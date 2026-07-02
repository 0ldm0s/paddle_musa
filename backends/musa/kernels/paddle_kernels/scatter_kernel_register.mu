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

// #include "paddle/phi/kernels/scatter_kernel.h"
// #include "paddle/phi/backends/gpu/gpu_context.h"
// #include "paddle/phi/core/kernel_registry.h"
// #include "paddle/phi/kernels/gpu/flash_attn_utils.h"

// #include <musa_runtime.h>

// using ::musa::dnn::Scatter;
// using ::musa::dnn::MemoryHandler;
// using muTensor = ::musa::dnn::Tensor;

// namespace phi {

// template <typename T, typename IndexT>
// __global__ void ScatterAddNoAtomicKernel(const IndexT* index,
//                                          const T* updates,
//                                          int64_t index_numel,
//                                          int64_t slice_size,
//                                          int64_t out_numel,
//                                          T* out) {
//   int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int64_t stride = blockDim.x * gridDim.x;

//   for (int64_t out_offset = tid; out_offset < out_numel; out_offset += stride) {
//     int64_t dst_index = out_offset / slice_size;
//     int64_t inner_offset = out_offset % slice_size;

//     T acc = out[out_offset];

//     for (int64_t i = 0; i < index_numel; ++i) {
//       int64_t idx = static_cast<int64_t>(index[i]);
//       if (idx == dst_index) {
//         acc = acc + updates[i * slice_size + inner_offset];
//       }
//     }

//     out[out_offset] = acc;
//   }
// }

// template <typename T, typename IndexT>
// __global__ void SlowScatterAddOnlyKernel(const IndexT* index,
//                                          const T* updates,
//                                          int64_t index_numel,
//                                          int64_t slice_size,
//                                          T* out) {
//   for (int64_t i = 0; i < index_numel; ++i) {
//     int64_t dst_index = static_cast<int64_t>(index[i]);

//     for (int64_t j = 0; j < slice_size; ++j) {
//       int64_t dst_offset = dst_index * slice_size + j;
//       int64_t src_offset = i * slice_size + j;
//       out[dst_offset] = out[dst_offset] + updates[src_offset];
//     }
//   }
// }

// template <typename T, typename Context>
// void ScatterKernel(const Context& dev_ctx,
//                    const DenseTensor& x,
//                    const DenseTensor& index,
//                    const DenseTensor& updates,
//                    bool overwrite,
//                    DenseTensor* out) {
//   if (index.numel() == 0) {
//     dev_ctx.template Alloc<T>(out);
//     phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
//     return;
//   }

//   if (out && out->numel() == 0) {
//     dev_ctx.template Alloc<T>(out);
//     return;
//   }

//   auto index_type = index.dtype();
//   bool index_type_match =
//       index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;

//   PADDLE_ENFORCE_EQ(
//       index_type_match,
//       true,
//       common::errors::InvalidArgument(
//           "scatter_op Index holds the wrong type, it holds [%s], "
//           "but desires to be [%s] or [%s].",
//           index_type,
//           phi::DataType::INT32,
//           phi::DataType::INT64));

//   dev_ctx.template Alloc<T>(out);

//   phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);

//   PADDLE_ENFORCE_GT(
//       index.numel(),
//       0,
//       common::errors::InvalidArgument("scatter index numel must be > 0."));

//   PADDLE_ENFORCE_EQ(
//       updates.numel() % index.numel(),
//       0,
//       common::errors::InvalidArgument(
//           "updates.numel() must be divisible by index.numel()."));

//   const int64_t index_numel = index.numel();
//   const int64_t slice_size = updates.numel() / index_numel;
//   const int64_t out_numel = out->numel();

//   if (overwrite) {
//     VLOG(1) << "using mudnn scatter overwrite=True";

//     auto& h = GetMudnnHandle<Context>(dev_ctx);

//     ::musa::dnn::Scatter mudnn_scatter;
//     CHECK_MUDNN_STATUS(
//         mudnn_scatter.SetMode(::musa::dnn::Scatter::Mode::UPDATE_ONLY),
//         "SetMode");

//     auto musa_out = CreateMUTensor(*out);
//     auto musa_x = CreateMUTensor(x);
//     auto musa_index = CreateMUTensor(index);
//     auto musa_updates = CreateMUTensor(updates);

//     auto place = dev_ctx.GetPlace();
//     ::musa::dnn::MemoryMaintainer maintainer =
//         [place](size_t bytes) { return PaddleInternalMemAlloc(bytes, place); };

//     constexpr int dim = 0;

//     CHECK_MUDNN_STATUS(
//         mudnn_scatter.Run(
//             h,
//             musa_out,
//             musa_x,
//             musa_index,
//             musa_updates,
//             dim,
//             maintainer),
//         "Run Mudnn Scatter.");
//   } else {
//     VLOG(1) << "using no-atomic scatter overwrite=False";

//     const T* updates_data = updates.data<T>();
//     T* out_data = out->data<T>();

//     constexpr int threads = 256;
//     int64_t blocks64 = (out_numel + threads - 1) / threads;
//     int blocks = static_cast<int>(blocks64 > 4096 ? 4096 : blocks64);

//     if (index_type == phi::DataType::INT32) {
//       const int32_t* index_data = index.data<int32_t>();
//       ScatterAddNoAtomicKernel<T, int32_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
//           index_data,
//           updates_data,
//           index_numel,
//           slice_size,
//           out_numel,
//           out_data);
//     } else {
//       const int64_t* index_data = index.data<int64_t>();
//       ScatterAddNoAtomicKernel<T, int64_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
//           index_data,
//           updates_data,
//           index_numel,
//           slice_size,
//           out_numel,
//           out_data);
//     }
//   }
// }

// }  // namespace phi

#include "paddle/phi/kernels/scatter_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/scatter.cu.h"

namespace phi {

template <typename T, typename IndexT>
__global__ void ScatterAddNoAtomicKernel(const IndexT* index,
                                         const T* updates,
                                         int64_t index_numel,
                                         int64_t slice_size,
                                         int64_t out_numel,
                                         T* out) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  int64_t output_count = out_numel / slice_size;

  for (int64_t out_offset = tid; out_offset < out_numel; out_offset += stride) {
    int64_t dst_index = out_offset / slice_size;
    int64_t inner_offset = out_offset % slice_size;

    T acc = static_cast<T>(0);
    bool matched = false;

    for (int64_t i = 0; i < index_numel; ++i) {
      int64_t idx = static_cast<int64_t>(index[i]);
      if (idx < 0) {
        idx += output_count;
      }

      if (idx == dst_index) {
        matched = true;
        acc = acc + updates[i * slice_size + inner_offset];
      }
    }

    if (matched) {
      out[out_offset] = acc;
    }
  }
}

template <typename T, typename Context>
void ScatterKernel(const Context &dev_ctx,
                   const DenseTensor &x,
                   const DenseTensor &index,
                   const DenseTensor &updates,
                   bool overwrite,
                   DenseTensor *out) {
  if (!overwrite) {
    if (index.numel() == 0) {
      dev_ctx.template Alloc<T>(out);
      phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
      return;
    }

    if (out && out->numel() == 0) {
      dev_ctx.template Alloc<T>(out);
      return;
    }

    auto index_type = index.dtype();
    bool index_type_match =
        index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;

    PADDLE_ENFORCE_EQ(
        index_type_match,
        true,
        common::errors::InvalidArgument(
            "scatter_op Index holds the wrong type, it holds [%s], "
            "but desires to be [%s] or [%s].",
            index_type,
            phi::DataType::INT32,
            phi::DataType::INT64));

    dev_ctx.template Alloc<T>(out);

    phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);

    PADDLE_ENFORCE_GT(
        index.numel(),
        0,
        common::errors::InvalidArgument("scatter index numel must be > 0."));

    PADDLE_ENFORCE_EQ(
        updates.numel() % index.numel(),
        0,
        common::errors::InvalidArgument(
            "updates.numel() must be divisible by index.numel()."));

    const int64_t index_numel = index.numel();
    const int64_t slice_size = updates.numel() / index_numel;
    const int64_t out_numel = out->numel();

    VLOG(1) << "using no-atomic scatter overwrite=False";

    const T* updates_data = updates.data<T>();
    T* out_data = out->data<T>();

    constexpr int threads = 256;
    int64_t blocks64 = (out_numel + threads - 1) / threads;
    int blocks = static_cast<int>(blocks64 > 4096 ? 4096 : blocks64);

    if (index_type == phi::DataType::INT32) {
      const int32_t* index_data = index.data<int32_t>();
      ScatterAddNoAtomicKernel<T, int32_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          index_data,
          updates_data,
          index_numel,
          slice_size,
          out_numel,
          out_data);
    } else {
      const int64_t* index_data = index.data<int64_t>();
      ScatterAddNoAtomicKernel<T, int64_t><<<blocks, threads, 0, dev_ctx.stream()>>>(
          index_data,
          updates_data,
          index_numel,
          slice_size,
          out_numel,
          out_data);
    }
    return;
  }
  
  if (index.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }
  Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  // use template class to support int32_t and int64_t
  auto index_type = index.dtype();
  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    common::errors::InvalidArgument(
                        "scatter_op Index holds the wrong type, it holds [%s],"
                        "but desires to be [%s] or [%s].",
                        index_type,
                        DataType::INT32,
                        DataType::INT64));
  if (index_type == DataType::INT32) {
    funcs::GPUScatterAssign<T, int32_t>(
        dev_ctx, updates, index, out, overwrite);
  } else {
    funcs::GPUScatterAssign<T, int64_t>(
        dev_ctx, updates, index, out, overwrite);
  }
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(scatter,
                          musa,
                          ALL_LAYOUT,
                          phi::ScatterKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::float16,
                          phi::bfloat16) {}