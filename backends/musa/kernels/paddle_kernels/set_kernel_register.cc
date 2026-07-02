// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
// Copyright (c) 2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
// - [register musa backend]

#include "paddle/phi/kernels/set_kernel.h"

#include <cstring>

#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/full_kernel.h"

namespace phi {

static int64_t ComputeRequiredStorageSize(const std::vector<int64_t>& dims,
                                          const std::vector<int64_t>& stride,
                                          int64_t offset) {
  int64_t required = offset;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] > 0) {
      required += (dims[i] - 1) * stride[i];
    }
  }
  return required + 1;
}

template <typename T, typename Context>
void SetKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& source,
               const std::vector<int64_t>& dims,
               const std::vector<int64_t>& stride,
               int64_t offset,
               DenseTensor* out) {
  auto meta = out->meta();
  meta.dims = DDim(dims.data(), static_cast<int>(dims.size()));
  meta.strides = DDim(stride.data(), static_cast<int>(stride.size()));
  meta.offset = offset;
  if (x.numel() == 0 || source.numel() == 0) {
    int64_t out_numel = 1;
    for (auto d : dims) {
      out_numel *= d;
    }
    if (source.numel() == 0 && x.numel() != 0) {
      if (out_numel == 0) {
        out->set_meta(meta);
        out->ShareInplaceVersionCounterWith(x);
        return;
      }
      int64_t required_size = ComputeRequiredStorageSize(dims, stride, offset);
      if (required_size > x.numel()) {
        DenseTensor tmp;
        std::vector<int64_t> alloc_shape = {required_size};
        Full<T, Context>(dev_ctx, alloc_shape, 0, &tmp);
        if (dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU) {
          std::memcpy(tmp.data<T>(), x.data<T>(), x.numel() * sizeof(T));
        } else {
          memory_utils::Copy(dev_ctx.GetPlace(),
                             tmp.data<T>(),
                             dev_ctx.GetPlace(),
                             x.data<T>(),
                             x.numel() * sizeof(T),
                             nullptr);
        }
        out->clear();
        *out = DenseTensor{tmp.Holder(), meta};
      } else {
        out->set_meta(meta);
      }
    } else if (source.numel() == 0 && x.numel() == 0 && out_numel != 0) {
      int64_t required_size = ComputeRequiredStorageSize(dims, stride, offset);
      DenseTensor tmp;
      std::vector<int64_t> alloc_shape = {required_size};
      Full<T, Context>(dev_ctx, alloc_shape, 0, &tmp);
      out->clear();
      *out = DenseTensor{tmp.Holder(), meta};
    } else if (source.numel() != 0) {
      out->clear();
      *out = DenseTensor{source.Holder(), meta};
    } else {
      out->clear();
      *out = DenseTensor{source.Holder(), meta};
    }
    out->ShareInplaceVersionCounterWith(x);
    return;
  }
  if (x.IsSharedWith(source)) {
    out->set_meta(meta);
  } else {
    out->clear();
    *out = DenseTensor{source.Holder(), meta};
  }
  out->ShareInplaceVersionCounterWith(x);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(set,
                          musa,
                          ALL_LAYOUT,
                          phi::SetKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16,
                          phi::complex64,
                          phi::complex128) {}
