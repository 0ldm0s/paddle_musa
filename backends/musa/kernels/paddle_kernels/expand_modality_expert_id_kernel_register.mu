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

#include "paddle/phi/kernels/legacy/gpu/expand_modality_expert_id_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void ExpandModalityExpertIdKernel(const T* expert_id,
                                             T* expert_id_out,
                                             int64_t numel,
                                             int64_t k,
                                             int64_t num_expert_per_modality,
                                             int64_t group_size,
                                             int64_t modality_offset,
                                             bool is_group_expert) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (int64_t i = idx; i < numel; i += blockDim.x * gridDim.x) {
    T e = expert_id[i];
    if (is_group_expert) {
      e += (i % k) * group_size;
    }
    if (num_expert_per_modality > 0) {
      T rank = e / num_expert_per_modality;
      T expert_id_in_rank = e % num_expert_per_modality;
      e = rank * (num_expert_per_modality * 2) + expert_id_in_rank +
          modality_offset * num_expert_per_modality;
    }
    expert_id_out[i] = e;
  }
}

template <typename T, typename Context>
void ExpandModalityExpertIDKernel(const Context& dev_ctx,
                                  const DenseTensor& expert_id,
                                  int64_t num_expert_per_modality,
                                  int64_t group_size,
                                  int64_t modality_offset,
                                  bool is_group_expert,
                                  DenseTensor* expert_id_out) {
  T* out_data = dev_ctx.template Alloc<T>(expert_id_out);
  const auto& dims = expert_id.dims();
  int64_t numel = expert_id.numel();
  if (numel <= 0) {
    return;
  }
  int64_t k = dims[dims.size() - 1];
  int block_size = std::min(256, dev_ctx.GetMaxThreadsPerBlock());
  int grid_size = static_cast<int>((numel + block_size - 1) / block_size);
  ExpandModalityExpertIdKernel<T><<<grid_size, block_size, 0, dev_ctx.stream()>>>(
      expert_id.data<T>(),
      out_data,
      numel,
      k,
      num_expert_per_modality,
      group_size,
      modality_offset,
      is_group_expert);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(expand_modality_expert_id,
                          musa,
                          ALL_LAYOUT,
                          phi::ExpandModalityExpertIDKernel,
                          int,
                          int64_t) {}
