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


#include "paddle/phi/kernels/slice_grad_kernel.h"
#include "paddle/phi/kernels/impl/slice_grad_kernel_impl.h"
#include "paddle/phi/kernels/sparse/unary_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(slice_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::SliceGradKernel,
                   bool,
                   int,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   int8_t,
                   phi::complex64,
                   phi::complex128,
                   phi::bfloat16,
                   phi::float16) {}

PD_CUSTOM_KERNEL_REGISTER(slice_array_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::SliceArrayGradKernel,
                   bool,
                   int,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   int8_t,
                   phi::complex64,
                   phi::complex128,
                   phi::bfloat16,
                   phi::float16) {}

PD_CUSTOM_KERNEL_REGISTER(slice_array_dense_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::SliceArrayDenseGradKernel,
                   bool,
                   int,
                   uint8_t,
                   int64_t,
                   float,
                   double,
                   int16_t,
                   int8_t,
                   phi::complex64,
                   phi::complex128,
                   phi::bfloat16,
                   phi::float16) {}

PD_CUSTOM_KERNEL_REGISTER(slice_coo_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::sparse::SliceCooGradKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_CUSTOM_KERNEL_REGISTER(slice_csr_grad,
                   musa,
                   ALL_LAYOUT,
                   phi::sparse::SliceCsrGradKernel,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
