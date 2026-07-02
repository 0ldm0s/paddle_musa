/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

// Modifications:
// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
// - [register musa backend]

#include "paddle/phi/kernels/softmax_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"
#include "paddle/phi/kernels/impl/softmax_kernel_impl.h"

#include "kernels/musa_context.h"
#include "kernels/kernels_utils.h"

#include <musa_runtime.h>

using ::musa::dnn::Softmax;
using ::musa::dnn::MemoryHandler;
using muTensor = ::musa::dnn::Tensor;

namespace phi {

template <typename T>
static bool UseMudnnDirectSoftmax(const DenseTensor& x, int axis) {
  const int rank = x.dims().size();
  const int calc_axis = axis < 0 ? axis + rank : axis;
  if (rank == 0 || calc_axis < 0 || calc_axis >= rank) {
    return false;
  }

  int64_t inner_size = 1;
  for (int i = calc_axis + 1; i < rank; ++i) {
    inner_size *= x.dims()[i];
  }
  const int64_t dim_size = x.dims()[calc_axis];
  constexpr int64_t io_bits = 128;
  const int64_t vec_len = io_bits / (sizeof(T) * 8);
  return inner_size == 1 && dim_size > 1 && dim_size <= 512 &&
         dim_size % vec_len != 0;
}

static bool UsePaddleNativeSoftmax(const DenseTensor& x, int axis) {
  const int rank = x.dims().size();
  const int calc_axis = axis < 0 ? axis + rank : axis;
  if (rank != 3 || calc_axis != 2) {
    return false;
  }
  return (x.dims()[0] == 4 && x.dims()[1] == 36864 && x.dims()[2] == 2) ||
         (x.dims()[0] == 2 && x.dims()[1] == 32768 && x.dims()[2] == 2);
}

template <typename T, typename Context>
void SoftmaxMudnnKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        int axis,
                        DenseTensor* out) {
  if (UsePaddleNativeSoftmax(x, axis)) {
    VLOG(1) << "using paddle native softmax for OCRNet attention shape";
    phi::SoftmaxKernel<T, Context>(dev_ctx, x, axis, out);
    return;
  }

  const auto softmax_algorithm = UseMudnnDirectSoftmax<T>(x, axis)
                                   ? ::musa::dnn::Softmax::Algorithm::DIRECT
                                   : ::musa::dnn::Softmax::Algorithm::ACCURATE;

  VLOG(1) << "using mudnn softmax";
  auto& h = GetMudnnHandle<Context>(dev_ctx);
  ::musa::dnn::Softmax ddnSoftmax;
  MUDNN_CHECK(ddnSoftmax.SetAlgorithm(softmax_algorithm), "SetAlgorithm");
  MUDNN_CHECK(ddnSoftmax.SetMode(::musa::dnn::Softmax::Mode::SOFTMAX), "SetMode");
  MUDNN_CHECK(ddnSoftmax.SetDim(axis), "SetDim");

  dev_ctx.template Alloc<T>(out);
  auto musa_out = CreateMUTensor(*out);
  auto musa_x = CreateMUTensor(x);

  auto place = dev_ctx.GetPlace();
  ::musa::dnn::MemoryMaintainer maintainer =
      [place](size_t bytes) { return PaddleInternalMemAlloc(bytes, place); };

  MUDNN_CHECK(
      ddnSoftmax.Run(
          h,
          musa_out,
          musa_x,
          maintainer),
      "Run Mudnn Softmax Fwd.");
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(softmax,
                          musa,
                          ALL_LAYOUT,
                          phi::SoftmaxMudnnKernel,
                          float,
                          phi::float16,
                          phi::bfloat16) {}
