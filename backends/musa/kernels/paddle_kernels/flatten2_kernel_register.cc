// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/flatten2_kernel_impl.h"

PD_CUSTOM_KERNEL_REGISTER(flatten2,
                          musa,
                          ALL_LAYOUT,
                          phi::Flatten2Kernel,
                          float,
                          double,
                          uint8_t,
                          int,
                          int8_t,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(flatten2_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::Flatten2GradKernel,
                          float,
                          double,
                          uint8_t,
                          int,
                          int8_t,
                          int64_t) {}
