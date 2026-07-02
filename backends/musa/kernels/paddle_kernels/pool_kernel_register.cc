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


#include "paddle/phi/kernels/pool_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MusaPool2dKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const IntArray& kernel_size,
                      const std::vector<int64_t>& strides,
                      const std::vector<int64_t>& paddings,
                      bool ceil_mode,
                      bool exclusive,
                      const std::string& data_format,
                      const std::string& pooling_type,
                      bool global_pooling,
                      bool adaptive,
                      const std::string& padding_algorithm,
                      DenseTensor* out) {
  if (adaptive && pooling_type == "avg" && !global_pooling &&
      data_format == "NCHW" && x.dims().size() == 4 &&
      out->dims().size() == 4 && out->dims()[2] == 1 && out->dims()[3] == 1) {
    MeanRawKernel<T, Context>(
        dev_ctx, x, IntArray(std::vector<int64_t>{2, 3}), true, false, out);
    return;
  }

  Pool2dKernel<T, Context>(dev_ctx,
                           x,
                           kernel_size,
                           strides,
                           paddings,
                           ceil_mode,
                           exclusive,
                           data_format,
                           pooling_type,
                           global_pooling,
                           adaptive,
                           padding_algorithm,
                           out);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(pool2d,
                   musa,
                   ALL_LAYOUT,
                   phi::MusaPool2dKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(lp_pool2d,
                   musa,
                   ALL_LAYOUT,
                   phi::LPPool2dKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(max_pool2d_with_index,
                   musa,
                   ALL_LAYOUT,
                   phi::MaxPool2dWithIndexKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}

PD_CUSTOM_KERNEL_REGISTER(pool3d,
                   musa,
                   ALL_LAYOUT,
                   phi::Pool3dKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(max_pool3d_with_index,
                   musa,
                   ALL_LAYOUT,
                   phi::MaxPool3dWithIndexKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}

PD_CUSTOM_KERNEL_REGISTER(fractional_max_pool2d,
                   musa,
                   ALL_LAYOUT,
                   phi::FractionalMaxPool2dKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}

PD_CUSTOM_KERNEL_REGISTER(fractional_max_pool3d,
                   musa,
                   ALL_LAYOUT,
                   phi::FractionalMaxPool3dKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::CppTypeToDataType<int>::Type());
}
