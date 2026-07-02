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

#include "paddle/phi/core/kernel_registry.h"
#define __CUDACC__ __MUSACC__
#define cudaGetLastError musaGetLastError
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#include "paddle/phi/kernels/gpu/rms_norm_cuda_kernel.h"
#undef cudaMemcpyDeviceToDevice
#undef cudaGetLastError
#undef __CUDACC__

PD_CUSTOM_KERNEL_REGISTER(rms_norm,
                          musa,
                          ALL_LAYOUT,
                          phi::RMSNormFwdKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(rms_norm_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::RMSNormBwdKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}
