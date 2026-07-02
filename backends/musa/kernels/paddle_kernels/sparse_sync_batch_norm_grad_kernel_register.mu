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
#include "paddle/phi/kernels/sync_batch_norm_grad_kernel.h"

namespace phi {
namespace sparse {

template <typename T, typename Context>
void EmptyLikeCooForSyncBatchNormGrad(const Context& dev_ctx,
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
void SyncBatchNormCooGradKernel(const Context& dev_ctx,
                                const SparseCooTensor& x,
                                const DenseTensor& scale,
                                const DenseTensor& bias,
                                const DenseTensor& saved_mean,
                                const DenseTensor& saved_variance,
                                const paddle::optional<DenseTensor>& reserve_space,
                                const SparseCooTensor& y_grad,
                                float momentum,
                                float epsilon,
                                const std::string& data_layout,
                                bool is_test,
                                bool use_global_stats,
                                bool trainable_statistics,
                                SparseCooTensor* x_grad,
                                DenseTensor* scale_grad,
                                DenseTensor* bias_grad) {
  EmptyLikeCooForSyncBatchNormGrad<T, Context>(dev_ctx, x, x_grad);
  scale_grad->Resize(scale.dims());
  bias_grad->Resize(bias.dims());
  dev_ctx.template Alloc<T>(scale_grad);
  dev_ctx.template Alloc<T>(bias_grad);
  phi::SyncBatchNormGradKernel<T, Context>(dev_ctx,
                                           x.values(),
                                           scale,
                                           bias,
                                           saved_mean,
                                           saved_variance,
                                           reserve_space,
                                           y_grad.values(),
                                           momentum,
                                           epsilon,
                                           data_layout,
                                           is_test,
                                           use_global_stats,
                                           trainable_statistics,
                                           x_grad->mutable_values(),
                                           scale_grad,
                                           bias_grad);
}

}  // namespace sparse
}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(sync_batch_norm_coo_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::sparse::SyncBatchNormCooGradKernel,
                          float,
                          phi::float16) {}
