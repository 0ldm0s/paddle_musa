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

#include "paddle/phi/kernels/sparse/batch_norm_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/batch_norm_grad_kernel.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"

namespace phi::sparse {

template <typename T, typename Context>
void EmptyLikeCooForBatchNorm(const Context& dev_ctx,
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
void BatchNormCooKernel(const Context& dev_ctx,
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
  EmptyLikeCooForBatchNorm<T, Context>(dev_ctx, x, y);
  phi::BatchNormKernel<T, Context>(dev_ctx,
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

template <typename T, typename Context>
void BatchNormCooGradKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const DenseTensor& scale,
                            const DenseTensor& bias,
                            const paddle::optional<DenseTensor>& mean,
                            const paddle::optional<DenseTensor>& variance,
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
  EmptyLikeCooForBatchNorm<T, Context>(dev_ctx, x, x_grad);

  PADDLE_ENFORCE_EQ((scale_grad == nullptr && bias_grad == nullptr) ||
                        (scale_grad != nullptr && bias_grad != nullptr),
                    true,
                    common::errors::InvalidArgument(
                        "Weight and bias's stop_gradient of BatchNorm must be "
                        "True or False at the same time."));

  if (scale_grad && bias_grad) {
    scale_grad->Resize(scale.dims());
    bias_grad->Resize(bias.dims());
    dev_ctx.template Alloc<T>(scale_grad);
    dev_ctx.template Alloc<T>(bias_grad);
  }
  phi::BatchNormGradKernel<T, Context>(dev_ctx,
                                       x.values(),
                                       scale,
                                       bias,
                                       mean,
                                       variance,
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

}  // namespace phi::sparse

PD_CUSTOM_KERNEL_REGISTER(batch_norm_coo,
                          musa,
                          ALL_LAYOUT,
                          phi::sparse::BatchNormCooKernel,
                          float,
                          phi::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);
  kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);
}

PD_CUSTOM_KERNEL_REGISTER(batch_norm_coo_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::sparse::BatchNormCooGradKernel,
                          float,
                          phi::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
