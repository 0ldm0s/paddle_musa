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


#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void FullBatchSizeLikeKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const std::vector<int>& shape UNUSED,
                             const Scalar& val,
                             DataType dtype,
                             int x_batch_size_dim,
                             int out_batch_size_dim,
                             DenseTensor* out) {
  if (!x.lod().empty() && x_batch_size_dim == 0) {
    // set the correct batch size for the DenseTensor.
    auto odims = out->dims();
    odims[out_batch_size_dim] = static_cast<int>(x.lod().back().size()) - 1;
    FullKernel<T, Context>(dev_ctx, common::vectorize(odims), val, dtype, out);
  }
  FullLikeKernel<T, Context>(dev_ctx, x, val, dtype, out);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(full,
                   musa,
                   ALL_LAYOUT,
                   phi::FullKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(full_like,
                   musa,
                   ALL_LAYOUT,
                   phi::FullLikeKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::float8_e4m3fn,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_CUSTOM_KERNEL_REGISTER(full_with_tensor,
                   musa,
                   ALL_LAYOUT,
                   phi::FullWithTensorKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {
  kernel->InputAt(0).SetBackend(phi::Backend::CPU);
}

PD_CUSTOM_KERNEL_REGISTER(full_batch_size_like,
                   musa,
                   ALL_LAYOUT,
                   phi::FullBatchSizeLikeKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}