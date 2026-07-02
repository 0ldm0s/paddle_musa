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

#include "hack/cuda_hack/common_porting.h"
#include <thrust/system/musa/execution_policy.h>
namespace thrust {
namespace cuda = musa;
}  // namespace thrust
#include "paddle/phi/kernels/sparse/gpu/sparse_utils_kernel.cu"
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(dense_to_coo, musa, ALL_LAYOUT, phi::sparse::DenseToCooKernel, float, double, phi::float16, uint8_t, int8_t, int16_t, int, int64_t, phi::complex64, phi::complex128) {}
PD_CUSTOM_KERNEL_REGISTER(csr_to_coo, musa, ALL_LAYOUT, phi::sparse::CsrToCooKernel, float, double, phi::float16, uint8_t, int8_t, int16_t, int, int64_t, bool, phi::complex64, phi::complex128) {}
PD_CUSTOM_KERNEL_REGISTER(coo_to_csr, musa, ALL_LAYOUT, phi::sparse::CooToCsrKernel, float, double, phi::float16, uint8_t, int8_t, int16_t, int, int64_t, bool, phi::complex64, phi::complex128) {}
PD_CUSTOM_KERNEL_REGISTER(dense_to_csr, musa, ALL_LAYOUT, phi::sparse::DenseToCsrKernel, float, double, phi::float16, uint8_t, int8_t, int16_t, int, int64_t, phi::complex64, phi::complex128) {}
PD_CUSTOM_KERNEL_REGISTER(coo_to_dense, musa, ALL_LAYOUT, phi::sparse::CooToDenseKernel, float, double, phi::float16, uint8_t, int8_t, int16_t, int, int64_t, bool, phi::complex64, phi::complex128) {}
PD_CUSTOM_KERNEL_REGISTER(csr_to_dense, musa, ALL_LAYOUT, phi::sparse::CsrToDenseKernel, float, double, phi::float16, uint8_t, int8_t, int16_t, int, int64_t, bool, phi::complex64, phi::complex128) {}
PD_CUSTOM_KERNEL_REGISTER(indices_coo, musa, ALL_LAYOUT, phi::sparse::IndicesCooKernel, float, double, phi::float16, uint8_t, int8_t, int16_t, int, int64_t) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}

PD_CUSTOM_KERNEL_REGISTER(values_coo, musa, ALL_LAYOUT, phi::sparse::ValuesCooKernel, float, double, phi::float16, uint8_t, int8_t, int16_t, int, int64_t, bool, phi::complex64, phi::complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
PD_CUSTOM_KERNEL_REGISTER(values_csr, musa, ALL_LAYOUT, phi::sparse::ValuesCsrKernel, float, double, phi::float16, uint8_t, int8_t, int16_t, int, int64_t, bool, phi::complex64, phi::complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
PD_CUSTOM_KERNEL_REGISTER(sparse_coo_tensor, musa, ALL_LAYOUT, phi::sparse::SparseCooTensorKernel, float, double, phi::float16, uint8_t, int16_t, int, int64_t, phi::complex64, phi::complex128) {}
