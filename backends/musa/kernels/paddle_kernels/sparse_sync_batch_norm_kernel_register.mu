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
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sync_batch_norm_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void EmptyLikeCooForSyncBatchNorm(const Context& dev_ctx,
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
void SyncBatchNormCooKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& mean,
                            const DenseTensor& variance,
                            const DenseTensor& scale,
                            const DenseTensor& bias,
                            bool is_test,
                            float momentum,
                            float epsilon,
                            const std::string& data_layout,
                            bool use_global_stats,
                            bool trainable_statistics,
                            SparseCooTensor* y,
                            DenseTensor* mean_out,
                            DenseTensor* variance_out,
                            DenseTensor* saved_mean,
                            DenseTensor* saved_variance,
                            DenseTensor* reserve_space) {
  EmptyLikeCooForSyncBatchNorm<T, Context>(dev_ctx, x, y);
  phi::SyncBatchNormKernel<T, Context>(dev_ctx,
                                       x.values(),
                                       mean,
                                       variance,
                                       scale,
                                       bias,
                                       is_test,
                                       momentum,
                                       epsilon,
                                       data_layout,
                                       use_global_stats,
                                       trainable_statistics,
                                       y->mutable_values(),
                                       mean_out,
                                       variance_out,
                                       saved_mean,
                                       saved_variance,
                                       reserve_space);
  y->SetIndicesDict(x.GetIndicesDict());
  y->SetKmaps(x.GetKmaps());
}

}  // namespace sparse
}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(sync_batch_norm_coo,
                          musa,
                          ALL_LAYOUT,
                          phi::sparse::SyncBatchNormCooKernel,
                          float,
                          phi::float16) {}
