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

#include "paddle/phi/kernels/fake_quantize_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void FakeQuantizeGradCopy(const Context& dev_ctx,
                          const DenseTensor& dout,
                          DenseTensor* dx) {
  PADDLE_ENFORCE_NOT_NULL(dx,
                          common::errors::PreconditionNotMet(
                              "The fake quantize grad output dx is nullptr"));
  dev_ctx.template Alloc<T>(dx);
  phi::Copy(dev_ctx, dout, dev_ctx.GetPlace(), false, dx);
}

template <typename T, typename Context>
void FakeChannelWiseQuantizeDequantizeAbsMaxGradMusaKernel(
    const Context& dev_ctx,
    const DenseTensor& dout,
    int bit_length,
    int round_type,
    int quant_axis,
    DenseTensor* dx) {
  FakeQuantizeGradCopy<T, Context>(dev_ctx, dout, dx);
}

template <typename T, typename Context>
void FakeQuantizeDequantizeAbsMaxGradMusaKernel(const Context& dev_ctx,
                                                const DenseTensor& dout,
                                                int bit_length,
                                                int round_type,
                                                DenseTensor* dx) {
  FakeQuantizeGradCopy<T, Context>(dev_ctx, dout, dx);
}

template <typename T, typename Context>
void FakeQuantizeDequantizeMovingAverageAbsMaxGradMusaKernel(
    const Context& dev_ctx,
    const DenseTensor& dout,
    float moving_rate,
    int bit_length,
    bool is_test,
    int round_type,
    DenseTensor* dx) {
  FakeQuantizeGradCopy<T, Context>(dev_ctx, dout, dx);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(fake_channel_wise_quantize_dequantize_abs_max_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::FakeChannelWiseQuantizeDequantizeAbsMaxGradMusaKernel,
                          float) {}

PD_CUSTOM_KERNEL_REGISTER(fake_quantize_dequantize_abs_max_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::FakeQuantizeDequantizeAbsMaxGradMusaKernel,
                          float,
                          phi::float16) {}

PD_CUSTOM_KERNEL_REGISTER(
    fake_quantize_dequantize_moving_average_abs_max_grad,
    musa,
    ALL_LAYOUT,
    phi::FakeQuantizeDequantizeMovingAverageAbsMaxGradMusaKernel,
    float,
    phi::float16) {}
