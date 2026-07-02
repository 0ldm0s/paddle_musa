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
#define __nv_fp8_e4m3 __mt_fp8_e4m3
#include "paddle/phi/kernels/legacy/gpu/fp8_quant_blockwise_kernel.cu"
#undef __nv_fp8_e4m3
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(fp8_quant_blockwise,
                          musa,
                          ALL_LAYOUT,
                          phi::FP8QuantBlockWiseKernel,
                          phi::float16,
                          phi::bfloat16,
                          float,
                          double) {}
