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

#include "paddle/phi/kernels/selected_rows/shape_kernel.cc"
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(shape_sr,
                          musa,
                          ALL_LAYOUT,
                          phi::sr::ShapeKernel,
                          bool,
                          int,
                          int8_t,
                          uint8_t,
                          int64_t,
                          float,
                          double,
                          phi::complex64,
                          phi::complex128) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT32);
}

PD_CUSTOM_KERNEL_REGISTER(shape64_sr,
                          musa,
                          ALL_LAYOUT,
                          phi::sr::Shape64Kernel,
                          bool,
                          int,
                          int8_t,
                          uint8_t,
                          int64_t,
                          float,
                          double,
                          phi::complex64,
                          phi::complex128) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->OutputAt(0).SetBackend(phi::Backend::CPU);
  kernel->OutputAt(0).SetDataType(phi::DataType::INT64);
}
