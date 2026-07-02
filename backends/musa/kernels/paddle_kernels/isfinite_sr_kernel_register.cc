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

#include "paddle/phi/kernels/selected_rows/isfinite_kernel.cc"
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(isinf_sr,
                          musa,
                          ALL_LAYOUT,
                          phi::IsinfSR,
                          float,
                          double,
                          phi::float16,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(isnan_sr,
                          musa,
                          ALL_LAYOUT,
                          phi::IsnanSR,
                          float,
                          double,
                          phi::float16,
                          int,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(isfinite_sr,
                          musa,
                          ALL_LAYOUT,
                          phi::IsfiniteSR,
                          float,
                          double,
                          phi::float16,
                          int,
                          int64_t) {}
