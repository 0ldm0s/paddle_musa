// Reserved. Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Modifications:
// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
// - [register musa backend]

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstdint> 

#include "glog/logging.h"  // For VLOG()
#include "paddle/phi/kernels/flash_attn_kernel.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/slice_kernel.h"
#include "paddle/utils/none.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h" 
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/tril_triu_kernel.h"

#include "kernels/kernels_utils.h"
#include "kernels/musa_context.h"

#include <musa_runtime.h>

using ::musa::dnn::BatchMatMul;
using ::musa::dnn::MemoryHandler;
using muTensor = ::musa::dnn::Tensor;


namespace phi {

muTensor CreateW4MUTensor(const DenseTensor& weight,
                          const bool trans, const std::string& weight_dtype) {
  muTensor w_m;
  // weight: (K // 8, M) -> w_m: (K, M) with INT4
  if (weight_dtype == "int4") {
    MUDNN_CHECK(w_m.SetType(muTensor::Type::QINT4), "Set QINT4 dtype");
  } else if (weight_dtype == "int8") {
    MUDNN_CHECK(w_m.SetType(muTensor::Type::QINT8), "Set QINT8 dtype");
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Only support int4 and int8 for weight_only_linear"));
  }
  MUDNN_CHECK(w_m.SetAddr(weight.data()), "SetAddr");
  MUDNN_CHECK(w_m.SetFormat(muTensor::Format::NCHW), "SetFormat");
  if (trans) {
    // set transposed shape and stride
    MUDNN_CHECK(w_m.SetNdInfo({weight.dims()[1], weight.dims()[0] * 8},
                                     {weight.strides()[1] * 8, weight.strides()[0]}),
                        "SetNdInfo");
  } else {
    MUDNN_CHECK(w_m.SetNdInfo({weight.dims()[0] * 8, weight.dims()[1]},
                                     {weight.strides()[0], weight.strides()[1]}),
                        "SetNdInfo");
  }
  return w_m;
}

static inline phi::DDim MakeContiguousStrides(const phi::DDim& dims) {
  const int n = dims.size();
  std::vector<int64_t> s(n, 1);

  int64_t acc = 1;
  for (int i = n - 1; i >= 0; --i) {
    s[i] = acc;
    acc *= dims[i];
  }
  return phi::make_ddim(s);
}

static inline bool IsTransposedLike(const phi::DenseTensor& t) {
  const auto& dims = t.dims();
  const auto& strides = t.strides();  // phi::DDim

  if (strides.size() != dims.size()) {
    return true;
  }

  phi::DDim expected = MakeContiguousStrides(dims);
  return !(strides == expected);
}

static inline int Rank(const phi::DenseTensor& t) {
  return static_cast<int>(t.dims().size());
}

static inline int64_t SizeAt(const phi::DenseTensor& t, int axis) {
  PADDLE_ENFORCE_GE(
      axis, 0,
      phi::errors::InvalidArgument("axis must be >= 0, got %d", axis));
  PADDLE_ENFORCE_LT(
      axis, Rank(t),
      phi::errors::InvalidArgument("axis %d out of range for rank %d", axis, Rank(t)));
  return t.dims()[axis];
}

static inline bool IsHalfOrBf16(phi::DataType dt) {
  return dt == phi::DataType::FLOAT16 || dt == phi::DataType::BFLOAT16;
}

static inline const char* DTypeName(phi::DataType dt) {
  switch (dt) {
    case phi::DataType::FLOAT16:  return "FLOAT16";
    case phi::DataType::BFLOAT16: return "BFLOAT16";
    case phi::DataType::FLOAT32:  return "FLOAT32";
    case phi::DataType::INT32:    return "INT32";
    case phi::DataType::INT64:    return "INT64";
    default:                      return "UNKNOWN";
  }
}

inline void CheckWeightOnlyLinearArgs(const phi::DenseTensor& in,
                                      const phi::DenseTensor& weight,
                                      const paddle::optional<phi::DenseTensor>& bias,
                                      const phi::DenseTensor& weight_scale,
                                      const std::string& weight_dtype,
                                      int32_t group_size) {
  PADDLE_ENFORCE_EQ(
      weight.dims().size(),
      2,
      phi::errors::InvalidArgument(
          "WeightOnlyLinear: weight must be 2-D, but got rank=%d, dims=%s.",
          weight.dims().size(),
          weight.dims()));

  PADDLE_ENFORCE_GE(
      in.dims().size(),
      1,
      phi::errors::InvalidArgument(
          "WeightOnlyLinear: input must have rank >=1, but got rank=%d, dims=%s.",
          in.dims().size(),
          in.dims()));

  int64_t in_k = in.dims()[in.dims().size() - 1];
  int64_t w_k  = weight.dims()[1];

  PADDLE_ENFORCE_EQ(
      w_k,
      in_k,
      phi::errors::InvalidArgument(
          "WeightOnlyLinear: weight.shape[1] must equal input.shape[-1]. "
          "But got weight.shape[1]=%ld, input.shape[-1]=%ld. "
          "weight dims=%s, input dims=%s.",
          w_k,
          in_k,
          weight.dims(),
          in.dims()));

  PADDLE_ENFORCE_EQ(
      weight_dtype,
      std::string("int4"),
      phi::errors::InvalidArgument(
          "WeightOnlyLinear: only support weight_dtype='int4', but got '%s'.",
          weight_dtype));

  // PADDLE_ENFORCE_EQ(
  //     group_size,
  //     -1,
  //     phi::errors::InvalidArgument(
  //         "WeightOnlyLinear: only support group_size=-1 currently, but got %d.",
  //         group_size));

  if (bias) {
    PADDLE_ENFORCE_GE(
        bias->dims().size(),
        1,
        phi::errors::InvalidArgument(
            "WeightOnlyLinear: bias rank must >=1, but got rank=%d dims=%s.",
            bias->dims().size(),
            bias->dims()));
  }
}

template <typename T, typename Context>
void WeightOnlyLinearKernel(const Context& dev_ctx,
                            const DenseTensor& in,
                            const DenseTensor& weight,
                            const paddle::optional<DenseTensor>& bias,
                            const DenseTensor& weight_scale,
                            const std::string& weight_dtype,
                            const int32_t arch,
                            const int32_t group_size,
                            DenseTensor* out) {
    dev_ctx.template Alloc<T>(out);
    if (out->numel() == 0 || in.numel() == 0 || weight.numel() == 0) {
        return;
    }
    CheckWeightOnlyLinearArgs(in, weight, bias, weight_scale, weight_dtype, group_size);

    auto place = dev_ctx.GetPlace();
    ::musa::dnn::MemoryMaintainer maintainer =
        [place](size_t bytes) { return PaddleInternalMemAlloc(bytes, place); };

    bool trans_l = false;
    bool trans_r = false;

    phi::DenseTensor weight_scale_2d = phi::Cast<T, Context>(dev_ctx, weight_scale, DataType::FLOAT32);
    weight_scale_2d.Resize({1, weight_scale.dims()[0]});

    auto& h = GetMudnnHandle<Context>(dev_ctx);
    ::musa::dnn::BatchMatMul op;
    
    muTensor in_mt = CreateMUTensor(in);
    muTensor o_mt = CreateMUTensor(*out);
    muTensor s_mt = CreateMUTensor(weight_scale_2d);

    DenseTensor transposed_weight;
    phi::DenseTensorMeta meta(
        weight.dtype(),
        phi::make_ddim({weight.dims()[1], weight.dims()[0]}),
        weight.layout());
    transposed_weight.set_meta(meta);
    std::vector<int> axis = {1, 0};
    phi::TransposeKernel<int8_t, Context>(dev_ctx, weight, axis, &transposed_weight);

    muTensor w_mt;
    w_mt.SetType(muTensor::Type::QINT4);
    const void* addr = transposed_weight.data();
    SetMUTensorAddr(addr, w_mt);
    w_mt.SetFormat(muTensor::Format::NCHW);
    w_mt.SetNdInfo({weight.dims()[1], weight.dims()[0] * 2},
                    {weight.dims()[0] * 2, 1});

    muTensor b_mt;
    if (bias.is_initialized()) {
        const auto& b = bias.get();
        b_mt = CreateMUTensor(b);
    } else {
        b_mt = muTensor();
    }

    MUDNN_CHECK(op.SetAlpha(1.0), "SetAlpha");
    MUDNN_CHECK(op.SetBeta(0.0), "SetBeta");
    MUDNN_CHECK(op.SetGamma(1.0), "SetGamma");
    MUDNN_CHECK(op.SetTranspose(trans_l, trans_r), "SetTranspose");
    MUDNN_CHECK(
        op.SetComputeMode(static_cast<::musa::dnn::MatMul::ComputeMode>(0)),
        "SetComputeMode");

    ::musa::dnn::MatMulLtParam param;
    MUDNN_CHECK(param.SetScale(muTensor(), s_mt, muTensor(), muTensor()),
                        "SetScale");
    MUDNN_CHECK(op.RunLt(h, o_mt, in_mt, w_mt, muTensor(), b_mt, param,
                                maintainer),
                        "RunLt");
}

}


PD_CUSTOM_KERNEL_REGISTER(weight_only_linear,
                   musa,
                   ALL_LAYOUT,
                   phi::WeightOnlyLinearKernel,
                   phi::float16,
                   phi::bfloat16) {}
