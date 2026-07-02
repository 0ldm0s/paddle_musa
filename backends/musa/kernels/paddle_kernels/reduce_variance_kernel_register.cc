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
#include "paddle/phi/kernels/reduce_variance_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/reduce_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const IntArray& dims,
                bool keep_dim,
                DenseTensor* out) {
  bool reduce_all = recompute_reduce_all(x, dims);
  if (std::is_same<T, int>::value || std::is_same<T, int64_t>::value ||
      std::is_same<T, bool>::value) {
    using Type = typename std::conditional<std::is_same<T, int>::value ||
                                               std::is_same<T, int64_t>::value ||
                                               std::is_same<T, bool>::value,
                                           float,
                                           T>::type;
    DenseTensor x_float = Cast<T, Context>(dev_ctx, x, phi::DataType::FLOAT32);
    DenseTensor out_float;
    out_float.Resize(out->dims());
    MeanRawKernel<Type>(dev_ctx, x_float, dims, keep_dim, reduce_all, &out_float);
    CastKernel<Type, Context>(dev_ctx, out_float, x.dtype(), out);
  } else {
    MeanRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
  }
}

template <typename T, typename Context>
void VarianceKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<int64_t>& dims,
                    bool keep_dim,
                    DenseTensor* out) {
  DenseTensor temp_mean = Mean<T, Context>(dev_ctx, x, dims, true);
  DenseTensor temp_differences = Subtract<T, Context>(dev_ctx, x, temp_mean);
  DenseTensor temp_pow =
      Multiply<T, Context>(dev_ctx, temp_differences, temp_differences);

  MeanKernel<T, Context>(dev_ctx, temp_pow, dims, keep_dim, out);
}
}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(
    variance, musa, ALL_LAYOUT, phi::VarianceKernel, float, double) {}
