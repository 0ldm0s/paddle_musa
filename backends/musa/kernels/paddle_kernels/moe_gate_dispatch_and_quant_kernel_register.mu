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
#include <musa_fp8.h>
#define cudaMemsetAsync musaMemsetAsync
#define __nv_fp8_e4m3 __mt_fp8_e4m3

namespace phi {
template <bool Power2Scaling>
__device__ __forceinline__ float RoundPower2Scale(float scale) {
  return scale;
}
}  // namespace phi

#include "paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_and_quant_kernel.cu"
#undef __nv_fp8_e4m3
#undef cudaMemsetAsync
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(moe_gate_dispatch_and_quant,
                          musa,
                          ALL_LAYOUT,
                          phi::MoeDispatchAndQuantKernel,
                          phi::bfloat16) {}
