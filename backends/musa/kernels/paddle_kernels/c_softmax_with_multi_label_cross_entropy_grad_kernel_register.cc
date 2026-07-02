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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void CSoftmaxWithMultiLabelCrossEntropyGradKernel(
    const Context& dev_ctx,
    const DenseTensor& softmax_in,
    const DenseTensor& label_in,
    const DenseTensor& smooth_weight_in,
    const DenseTensor& loss_grad_in,
    int64_t ignore_index,
    bool sum_multi_label_loss,
    int rank,
    int nranks,
    DenseTensor* logits_grad);

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(c_softmax_with_multi_label_cross_entropy_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::CSoftmaxWithMultiLabelCrossEntropyGradKernel,
                          float,
                          double,
                          phi::float16) {}
