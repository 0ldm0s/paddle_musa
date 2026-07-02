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

#include "paddle/phi/kernels/sparse/empty_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi::sparse {

template <typename T, typename Context>
void EmptyLikeCooKernel(const Context& dev_ctx,
                        const SparseCooTensor& x,
                        SparseCooTensor* out) {
  *(out->mutable_indices()) = x.indices();

  const DenseTensor& x_values = x.values();
  DenseTensor* out_values = out->mutable_values();
  out_values->Resize(x_values.dims());
  out->set_meta(x.meta());
  dev_ctx.template Alloc<T>(out_values);
}

template <typename T, typename Context>
void EmptyLikeCsrKernel(const Context& dev_ctx,
                        const SparseCsrTensor& x,
                        SparseCsrTensor* out) {
  *(out->mutable_crows()) = x.crows();
  *(out->mutable_cols()) = x.cols();

  const DenseTensor& x_values = x.values();
  DenseTensor* out_values = out->mutable_values();
  out_values->Resize(x_values.dims());
  out->set_meta(x.meta());
  dev_ctx.template Alloc<T>(out_values);
}

#define INSTANTIATE_EMPTY_LIKE_COO_KERNEL(T)                         \
  template void EmptyLikeCooKernel<T, phi::CustomContext>(           \
      const phi::CustomContext&, const SparseCooTensor&, SparseCooTensor*)

#define INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(T)                         \
  template void EmptyLikeCsrKernel<T, phi::CustomContext>(           \
      const phi::CustomContext&, const SparseCsrTensor&, SparseCsrTensor*)

INSTANTIATE_EMPTY_LIKE_COO_KERNEL(phi::float16);
INSTANTIATE_EMPTY_LIKE_COO_KERNEL(float);
INSTANTIATE_EMPTY_LIKE_COO_KERNEL(double);
INSTANTIATE_EMPTY_LIKE_COO_KERNEL(int8_t);
INSTANTIATE_EMPTY_LIKE_COO_KERNEL(uint8_t);
INSTANTIATE_EMPTY_LIKE_COO_KERNEL(int16_t);
INSTANTIATE_EMPTY_LIKE_COO_KERNEL(int);
INSTANTIATE_EMPTY_LIKE_COO_KERNEL(int64_t);
INSTANTIATE_EMPTY_LIKE_COO_KERNEL(bool);
INSTANTIATE_EMPTY_LIKE_COO_KERNEL(phi::complex64);
INSTANTIATE_EMPTY_LIKE_COO_KERNEL(phi::complex128);

INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(phi::float16);
INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(float);
INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(double);
INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(int8_t);
INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(uint8_t);
INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(int16_t);
INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(int);
INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(int64_t);
INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(bool);
INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(phi::complex64);
INSTANTIATE_EMPTY_LIKE_CSR_KERNEL(phi::complex128);

#undef INSTANTIATE_EMPTY_LIKE_COO_KERNEL
#undef INSTANTIATE_EMPTY_LIKE_CSR_KERNEL

}  // namespace phi::sparse

PD_CUSTOM_KERNEL_REGISTER(empty_like_coo, musa, ALL_LAYOUT, phi::sparse::EmptyLikeCooKernel, phi::float16, float, double, int8_t, uint8_t, int16_t, int, int64_t, bool, phi::complex64, phi::complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
}
PD_CUSTOM_KERNEL_REGISTER(empty_like_csr, musa, ALL_LAYOUT, phi::sparse::EmptyLikeCsrKernel, phi::float16, float, double, int8_t, uint8_t, int16_t, int, int64_t, bool, phi::complex64, phi::complex128) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR);
}
