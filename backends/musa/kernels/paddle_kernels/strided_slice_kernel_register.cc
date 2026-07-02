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


#include "paddle/phi/kernels/strided_slice_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void StridedSliceKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<int>& axes,
                        const IntArray& starts,
                        const IntArray& ends,
                        const IntArray& strides,
                        DenseTensor* out) {
  std::vector<int> infer_flags(axes.size(), 1);
  std::vector<int> decrease_axis;
  StridedSliceRawKernel<T, Context>(
      dev_ctx, x, axes, starts, ends, strides, infer_flags, decrease_axis, out);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(strided_slice,
                   musa,
                   ALL_LAYOUT,
                   phi::StridedSliceKernel,
                   float,
                   double,
                   bool,
                   int64_t,
                   int16_t,
                   int,
                   uint8_t,
                   int8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(strided_slice_raw,
                   musa,
                   ALL_LAYOUT,
                   phi::StridedSliceRawKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(strided_slice_array,
                   musa,
                   ALL_LAYOUT,
                   phi::StridedSliceArrayKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
