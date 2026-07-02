// Copyright (c) 2026 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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

#include "paddle/phi/kernels/sparse/sparse_utils_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/mask_kernel.h"

namespace phi::sparse {

template <typename T, typename Context>
void ValuesCooGradKernel(const Context& dev_ctx UNUSED,
                         const SparseCooTensor& x,
                         const DenseTensor& out_grad,
                         SparseCooTensor* x_grad) {
  x_grad->SetMember(x.indices(), out_grad, x.dims(), true);
}

template <typename T, typename Context>
void CooToDenseGradKernel(const Context& dev_ctx,
                          const SparseCooTensor& x,
                          const DenseTensor& out_grad,
                          SparseCooTensor* x_grad) {
  MaskAsCooKernel<T, Context>(dev_ctx, out_grad, x, x_grad);
}

}  // namespace phi::sparse

PD_CUSTOM_KERNEL_REGISTER(values_coo_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::sparse::ValuesCooGradKernel,
                          float,
                          double,
                          phi::float16,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          phi::complex64,
                          phi::complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_CUSTOM_KERNEL_REGISTER(coo_to_dense_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::sparse::CooToDenseGradKernel,
                          float,
                          double,
                          phi::float16,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          phi::complex64,
                          phi::complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_CUSTOM_KERNEL_REGISTER(sparse_coo_tensor_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::sparse::SparseCooTensorGradKernel,
                          float,
                          double,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          phi::complex64,
                          phi::complex128) {
  kernel->InputAt(1).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
