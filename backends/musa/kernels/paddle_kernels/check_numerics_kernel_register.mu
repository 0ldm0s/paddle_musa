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

#include <float.h>
#include <cmath>
#include <string>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/check_numerics_kernel.h"

namespace phi {

template <typename T>
__global__ void CheckNumericsStatsKernel(const T* x,
                                         int64_t numel,
                                         int64_t* stats,
                                         float* values) {
  __shared__ int64_t nan_buf[1024];
  __shared__ int64_t inf_buf[1024];
  __shared__ int64_t zero_buf[1024];
  __shared__ float max_buf[1024];
  __shared__ float min_buf[1024];
  __shared__ float sum_buf[1024];

  const int tid = threadIdx.x;
  int64_t num_nan = 0;
  int64_t num_inf = 0;
  int64_t num_zero = 0;
  float max_value = -FLT_MAX;
  float min_value = FLT_MAX;
  float sum_value = 0.0f;

  for (int64_t i = tid; i < numel; i += blockDim.x) {
    float value = static_cast<float>(x[i]);
    if (isnan(value)) {
      ++num_nan;
    } else if (isinf(value)) {
      ++num_inf;
    } else {
      max_value = value > max_value ? value : max_value;
      min_value = value < min_value ? value : min_value;
      sum_value += value;
    }
    if (value == 0.0f) {
      ++num_zero;
    }
  }

  nan_buf[tid] = num_nan;
  inf_buf[tid] = num_inf;
  zero_buf[tid] = num_zero;
  max_buf[tid] = max_value;
  min_buf[tid] = min_value;
  sum_buf[tid] = sum_value;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      nan_buf[tid] += nan_buf[tid + stride];
      inf_buf[tid] += inf_buf[tid + stride];
      zero_buf[tid] += zero_buf[tid + stride];
      max_buf[tid] = max_buf[tid] > max_buf[tid + stride] ? max_buf[tid]
                                                            : max_buf[tid + stride];
      min_buf[tid] = min_buf[tid] < min_buf[tid + stride] ? min_buf[tid]
                                                            : min_buf[tid + stride];
      sum_buf[tid] += sum_buf[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    stats[0] = nan_buf[0];
    stats[1] = inf_buf[0];
    stats[2] = zero_buf[0];
    values[0] = max_buf[0];
    values[1] = min_buf[0];
    values[2] = sum_buf[0] / static_cast<float>(numel);
  }
}

template <typename T, typename Context>
void CheckNumericsKernel(const Context& dev_ctx,
                         const DenseTensor& tensor,
                         const std::string& op_type UNUSED,
                         const std::string& var_name UNUSED,
                         const int check_nan_inf_level UNUSED,
                         const int stack_height_limit UNUSED,
                         const std::string& output_dir UNUSED,
                         DenseTensor* stats,
                         DenseTensor* values) {
  stats->Resize({3});
  values->Resize({3});
  int64_t* stats_data = dev_ctx.template Alloc<int64_t>(stats);
  float* values_data = dev_ctx.template Alloc<float>(values);

  if (tensor.numel() <= 0) {
    return;
  }

  CheckNumericsStatsKernel<T><<<1, 1024, 0, dev_ctx.stream()>>>(
      tensor.data<T>(), tensor.numel(), stats_data, values_data);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(check_numerics,
                          musa,
                          ALL_LAYOUT,
                          phi::CheckNumericsKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}
