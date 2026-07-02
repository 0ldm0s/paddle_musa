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
// Copyright (c) 2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
// - [register musa backend]

#include "paddle/phi/kernels/array_grad_kernel.h"

#include "paddle/common/layout.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/concat_grad_kernel.h"
#include "paddle/phi/kernels/stack_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void TensorToArrayKernel(const Context& dev_ctx,
                         const TensorArray& x,
                         const DenseTensor& out_grad,
                         int axis,
                         bool use_stack,
                         TensorArray* x_grad) {
  std::vector<DenseTensor> tmp_inputs(x.size());
  std::vector<const DenseTensor*> inputs;

  std::vector<DenseTensor*> inputs_grad;
  std::vector<DenseTensor> tmp_inputs_grad(x.size());

  for (size_t i = 0; i < x.size(); i++) {
    tmp_inputs[i].ShareDataWith(x[i]);
    inputs.push_back(&tmp_inputs[i]);
    inputs_grad.push_back(&tmp_inputs_grad[i]);
    inputs_grad[i]->set_meta(x[i].meta());
  }

  if (use_stack) {
    StackGradKernel<T, Context>(dev_ctx, out_grad, axis, inputs_grad);
  } else {
    ConcatGradKernel<T, Context>(dev_ctx, inputs, out_grad, axis, inputs_grad);
  }

  for (size_t i = 0; i < tmp_inputs_grad.size(); i++) {
    inputs_grad[i] = nullptr;
    x_grad->push_back(tmp_inputs_grad[i]);
  }
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(tensor_to_array,
                          musa,
                          ALL_LAYOUT,
                          phi::TensorToArrayKernel,
                          bool,
                          int,
                          int64_t,
                          float,
                          double,
                          phi::float16,
                          phi::complex64,
                          phi::complex128) {}
