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
#include "paddle/phi/kernels/gpu/nanmedian_kernel.cu"
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(nanmedian,
                          musa,
                          ALL_LAYOUT,
                          phi::NanmedianKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::float16,
                          phi::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
