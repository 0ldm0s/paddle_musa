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

#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SyncCalcStreamKernel(const Context& dev_ctx,
                          const DenseTensor& x UNUSED,
                          DenseTensor* out UNUSED) {
  phi::backends::gpu::GpuStreamSync(dev_ctx.stream());
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(sync_calc_stream,
                          musa,
                          ALL_LAYOUT,
                          phi::SyncCalcStreamKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::float16,
                          phi::bfloat16) {}
