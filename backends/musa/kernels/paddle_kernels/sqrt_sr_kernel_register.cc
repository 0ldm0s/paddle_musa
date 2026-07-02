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

#include "paddle/phi/kernels/selected_rows/activation_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void SqrtKernel(const Context& dev_ctx,
                const SelectedRows& x,
                SelectedRows* out) {
  out->set_rows(x.rows());
  out->set_height(x.height());
  phi::SqrtKernel<T, Context>(dev_ctx, x.value(), out->mutable_value());
}

}  // namespace sr
}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(
    sqrt_sr, musa, ALL_LAYOUT, phi::sr::SqrtKernel, float, double) {}
