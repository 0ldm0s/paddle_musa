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
#include "paddle/phi/kernels/sparse/gpu/conv_kernel.cu"
#include "paddle/phi/core/kernel_registry.h"

PD_CUSTOM_KERNEL_REGISTER(conv3d_coo,
                          musa,
                          ALL_LAYOUT,
                          phi::sparse::Conv3dCooKernel,
                          float,
                          double,
                          phi::float16) {
  kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO);
  kernel->OutputAt(0).SetDataType(paddle::DataType::UNDEFINED);
  kernel->OutputAt(1).SetDataType(paddle::DataType::INT32);
  kernel->OutputAt(2).SetDataType(paddle::DataType::INT32);
}
