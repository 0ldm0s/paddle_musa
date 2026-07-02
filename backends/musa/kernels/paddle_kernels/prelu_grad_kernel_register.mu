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
// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
// - [register musa backend]

#include "paddle/phi/kernels/prelu_grad_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

namespace {

constexpr int kPReluGradThreads = 256;

__global__ void PReluChannelFirstGradKernel(const float* x,
                                            const float* alpha,
                                            const float* out_grad,
                                            int64_t n,
                                            int64_t c,
                                            int64_t plane,
                                            float* x_grad,
                                            float* alpha_grad) {
  const int64_t channel = blockIdx.x;
  double sum = 0.0;

  for (int64_t offset = threadIdx.x; offset < n * plane;
       offset += blockDim.x) {
    const int64_t batch = offset / plane;
    const int64_t spatial = offset - batch * plane;
    const int64_t index = (batch * c + channel) * plane + spatial;
    const float x_val = x[index];
    const float dy = out_grad[index];
    if (x_grad != nullptr) {
      x_grad[index] = x_val > 0.0f ? dy : alpha[channel] * dy;
    }
    if (alpha_grad != nullptr && x_val <= 0.0f) {
      sum += static_cast<double>(x_val) * static_cast<double>(dy);
    }
  }

  __shared__ double shared[kPReluGradThreads];
  shared[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && alpha_grad != nullptr) {
    alpha_grad[channel] = static_cast<float>(shared[0]);
  }
}

}  // namespace

template <typename T, typename Context>
struct MusaPReluGradKernelImpl {
  static void Compute(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& alpha,
                      const DenseTensor& out_grad,
                      const std::string& data_format,
                      const std::string& mode,
                      DenseTensor* x_grad,
                      DenseTensor* alpha_grad) {
    PReluGradKernel<T, Context>(
        dev_ctx, x, alpha, out_grad, data_format, mode, x_grad, alpha_grad);
  }
};

template <typename Context>
struct MusaPReluGradKernelImpl<float, Context> {
  static void Compute(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& alpha,
                      const DenseTensor& out_grad,
                      const std::string& data_format,
                      const std::string& mode,
                      DenseTensor* x_grad,
                      DenseTensor* alpha_grad) {
    const auto& dims = x.dims();
    const bool use_channel_first_fast_path =
        mode == "channel" && data_format != "NHWC" && dims.size() >= 3 &&
        alpha.numel() == dims[1];

    if (!use_channel_first_fast_path) {
      PReluGradKernel<float, Context>(
          dev_ctx, x, alpha, out_grad, data_format, mode, x_grad, alpha_grad);
      return;
    }

    float* x_grad_ptr =
        x_grad ? dev_ctx.template Alloc<float>(x_grad) : nullptr;
    float* alpha_grad_ptr =
        alpha_grad ? dev_ctx.template Alloc<float>(alpha_grad) : nullptr;
    if (x.numel() == 0) {
      return;
    }
    if (x_grad_ptr == nullptr && alpha_grad_ptr == nullptr) {
      return;
    }

    int64_t plane = 1;
    for (int i = 2; i < dims.size(); ++i) {
      plane *= dims[i];
    }
    const int64_t n = dims[0];
    const int64_t c = dims[1];

    PReluChannelFirstGradKernel<<<c, kPReluGradThreads, 0, dev_ctx.stream()>>>(
        x.data<float>(),
        alpha.data<float>(),
        out_grad.data<float>(),
        n,
        c,
        plane,
        x_grad_ptr,
        alpha_grad_ptr);
  }
};

template <typename T, typename Context>
void MusaPReluGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& alpha,
                         const DenseTensor& out_grad,
                         const std::string& data_format,
                         const std::string& mode,
                         DenseTensor* x_grad,
                         DenseTensor* alpha_grad) {
  MusaPReluGradKernelImpl<T, Context>::Compute(
      dev_ctx, x, alpha, out_grad, data_format, mode, x_grad, alpha_grad);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(prelu_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::MusaPReluGradKernel,
                          float,
                          phi::float16,
                          phi::bfloat16,
                          double) {}
