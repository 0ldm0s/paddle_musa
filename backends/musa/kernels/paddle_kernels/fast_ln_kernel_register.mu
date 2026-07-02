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
#define nv_bfloat16 __mt_bfloat16
#define nv_bfloat162 mt_bfloat162
#define cudaDeviceProp musaDeviceProp
#define cudaError_t musaError_t
#define cudaError musaError
#define cudaSuccess musaSuccess
#define cudaGetErrorString musaGetErrorString
#define cudaGetDeviceProperties musaGetDeviceProperties
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor \
  musaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaFuncSetAttribute musaFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
  musaFuncAttributeMaxDynamicSharedMemorySize
#define cudaLaunchCooperativeKernel musaLaunchCooperativeKernel
#include "paddle/phi/kernels/legacy/gpu/ln.cu"
#include "paddle/phi/kernels/legacy/gpu/ln_fwd_cuda_kernel.cu"
#include "paddle/phi/kernels/legacy/gpu/ln_bwd_semi_cuda_kernel.cu"
#include "paddle/phi/kernels/legacy/gpu/fast_layernorm_kernel.cu"
#include "paddle/phi/kernels/legacy/gpu/fast_layernorm_grad_kernel.cu"
#include "paddle/phi/kernels/legacy/gpu/fast_rmsnorm_kernel.cu"
#include "paddle/phi/kernels/legacy/gpu/fast_rmsnorm_grad_kernel.cu"
#undef cudaLaunchCooperativeKernel
#undef cudaFuncAttributeMaxDynamicSharedMemorySize
#undef cudaFuncSetAttribute
#undef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#undef cudaGetDeviceProperties
#undef cudaGetErrorString
#undef cudaSuccess
#undef cudaError
#undef cudaError_t
#undef cudaDeviceProp
#undef nv_bfloat162
#undef nv_bfloat16
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(fast_ln,
                          musa,
                          ALL_LAYOUT,
                          phi::LnFwdKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(fast_ln_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::LnBwdKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(fast_rms_norm,
                          musa,
                          ALL_LAYOUT,
                          phi::RMSLnFwdKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(fast_rms_norm_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::RMSLnBwdKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}
