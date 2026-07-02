// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <climits>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/index_fill_util.h"
#include "paddle/phi/kernels/index_fill_grad_kernel.h"
#include "paddle/phi/kernels/index_fill_kernel.h"

namespace phi {

template <typename T, typename IndexT, typename IndT>
__global__ void IndexFillCudaKernel(const T* x UNUSED,
                                    const IndT* index,
                                    const IndexT index_size,
                                    const int dim UNUSED,
                                    const IndexT outer_size,
                                    const int64_t dim_size,
                                    const IndexT inner_size,
                                    const T fill_value,
                                    T* out) {
  IndexT idx =
      static_cast<IndexT>(threadIdx.x) +
      static_cast<IndexT>(blockIdx.x) * static_cast<IndexT>(blockDim.x);
  IndexT total = index_size * outer_size * inner_size;
  if (idx >= total) return;

  IndexT inner_idx = idx % inner_size;
  IndexT temp = idx / inner_size;
  IndexT index_idx = temp % index_size;
  IndexT outer_idx = temp / index_size;

  int64_t dim_idx = static_cast<int64_t>(index[index_idx]);
  if (dim_idx < 0) dim_idx += dim_size;

  if (dim_idx < 0 || dim_idx >= dim_size) return;

  IndexT offset = outer_idx * static_cast<IndexT>(dim_size) * inner_size +
                  static_cast<IndexT>(dim_idx) * inner_size + inner_idx;

  out[offset] = fill_value;
}

template <typename T, typename Context, typename IndexT, typename IndT>
void LaunchIndexFillCudaKernelImpl(const Context& dev_ctx,
                                   const T* x_data,
                                   const IndT* index_data,
                                   IndexT index_size,
                                   int dim,
                                   IndexT outer_size,
                                   int64_t dim_size,
                                   IndexT inner_size,
                                   T fill_value,
                                   T* out_data) {
  IndexT numel = outer_size * index_size * inner_size;
  auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  IndexFillCudaKernel<T, IndexT, IndT>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          x_data,
          index_data,
          index_size,
          dim,
          outer_size,
          dim_size,
          inner_size,
          fill_value,
          out_data);
}

template <typename T, typename Context, typename IndT>
void LaunchIndexFillCudaKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               int dim,
                               const DenseTensor& index,
                               const Scalar& value,
                               DenseTensor* out) {
  auto* x_data = x.data<T>();
  T fill_value = value.to<T>();

  bool is_initialized = out->initialized();
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (!is_initialized || (x.data<T>() != out->data<T>())) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
  }

  const IndT* index_data = index.data<IndT>();
  int64_t index_size = index.numel();

  if (index_size == 0) {
    return;
  }

  auto x_dims = x.dims();
  const int rank = x_dims.size();

  if (dim < 0) {
    dim += rank;
  }

  int64_t outer_size = 1;
  int64_t inner_size = 1;
  int64_t dim_size = x_dims[dim];

  for (int i = 0; i < dim; ++i) {
    outer_size *= x_dims[i];
  }
  for (int i = dim + 1; i < rank; ++i) {
    inner_size *= x_dims[i];
  }

  const int64_t numel = x.numel();
  constexpr int64_t kInt32Max = static_cast<int64_t>(INT32_MAX);

  if (numel <= kInt32Max) {
    LaunchIndexFillCudaKernelImpl<T, Context, int32_t, IndT>(
        dev_ctx,
        x_data,
        index_data,
        static_cast<int32_t>(index_size),
        dim,
        static_cast<int32_t>(outer_size),
        dim_size,
        static_cast<int32_t>(inner_size),
        fill_value,
        out_data);
  } else {
    LaunchIndexFillCudaKernelImpl<T, Context, int64_t, IndT>(dev_ctx,
                                                             x_data,
                                                             index_data,
                                                             index_size,
                                                             dim,
                                                             outer_size,
                                                             dim_size,
                                                             inner_size,
                                                             fill_value,
                                                             out_data);
  }
}

template <typename T, typename Context>
void IndexFillKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& index,
                     int dim,
                     const Scalar& value,
                     DenseTensor* out) {
  if (out && out->numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  auto x_dims = x.dims();
  const int rank = x_dims.size();

  int real_dim = dim;
  if (real_dim < 0) {
    real_dim += rank;
  }

  PADDLE_ENFORCE_GE(real_dim,
                    0,
                    common::errors::InvalidArgument(
                        "The dim must be >= -%d and < %d, but received %d.",
                        rank,
                        rank,
                        dim));
  PADDLE_ENFORCE_LT(real_dim,
                    rank,
                    common::errors::InvalidArgument(
                        "The dim must be >= -%d and < %d, but received %d.",
                        rank,
                        rank,
                        dim));

  PADDLE_ENFORCE_EQ(index.dims().size(),
                    1,
                    common::errors::InvalidArgument(
                        "The index tensor must be 1-D, but received %d-D.",
                        index.dims().size()));

  if (index.numel() == 0) {
    Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    return;
  }

  if (index.dtype() == DataType::INT32) {
    LaunchIndexFillCudaKernel<T, Context, int32_t>(
        dev_ctx, x, real_dim, index, value, out);
  } else if (index.dtype() == DataType::INT64) {
    LaunchIndexFillCudaKernel<T, Context, int64_t>(
        dev_ctx, x, real_dim, index, value, out);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of index must be int32 or int64, but received %s.",
        DataTypeToString(index.dtype())));
  }
}

template <typename T>
__global__ void IndexFillGradCudaKernel(const int64_t* index,
                                        const int64_t index_size,
                                        const int64_t dim_size,
                                        const int64_t outer_size,
                                        const int64_t inner_size,
                                        T* x_grad) {
  int64_t idx =
      static_cast<int64_t>(threadIdx.x) +
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blockIdx.x);
  int64_t total = index_size * outer_size * inner_size;

  if (idx >= total) {
    return;
  }

  int64_t inner_idx = idx % inner_size;
  int64_t temp = idx / inner_size;
  int64_t index_idx = temp % index_size;
  int64_t outer_idx = temp / index_size;

  int64_t dim_idx = index[index_idx];
  if (dim_idx < 0) {
    dim_idx += dim_size;
  }

  if (dim_idx < 0 || dim_idx >= dim_size) {
    return;
  }

  int64_t offset =
      outer_idx * dim_size * inner_size + dim_idx * inner_size + inner_idx;

  *(x_grad + offset) = static_cast<T>(0);
}

template <typename T, typename Context>
void LaunchIndexFillGradCudaKernel(const Context& dev_ctx,
                                   const DenseTensor& index,
                                   const DenseTensor& out_grad,
                                   const int dim,
                                   DenseTensor* x_grad) {
  Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);

  auto out_grad_dims = out_grad.dims();
  const int rank = out_grad_dims.size();

  DenseTensor index_int64;
  const DenseTensor* ptr_index = nullptr;

  if (index.dtype() == DataType::INT32) {
    index_int64.Resize(index.dims());
    dev_ctx.template Alloc<int64_t>(&index_int64);

    int64_t index_numel = index.numel();
    auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, index_numel);

    funcs::CastToInt64Kernel<int32_t><<<config.block_per_grid,
                                        config.thread_per_block,
                                        0,
                                        dev_ctx.stream()>>>(
        index.data<int32_t>(), index_int64.data<int64_t>(), index_numel);

    ptr_index = &index_int64;
  } else if (index.dtype() == DataType::INT64) {
    ptr_index = &index;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of index must be int32 or int64, but received %s.",
        DataTypeToString(index.dtype())));
  }

  const int64_t* index_data = ptr_index->data<int64_t>();
  int64_t index_size = ptr_index->numel();

  if (index_size == 0) {
    return;
  }

  int64_t outer_size = 1;
  int64_t inner_size = 1;
  int64_t dim_size = out_grad_dims[dim];

  for (int i = 0; i < dim; ++i) {
    outer_size *= out_grad_dims[i];
  }
  for (int i = dim + 1; i < rank; ++i) {
    inner_size *= out_grad_dims[i];
  }

  int64_t numel = outer_size * index_size * inner_size;
  auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);

  T* x_grad_data = x_grad->data<T>();

  IndexFillGradCudaKernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          index_data,
          index_size,
          dim_size,
          outer_size,
          inner_size,
          x_grad_data);
}

template <typename T, typename Context>
void IndexFillGradKernel(const Context& dev_ctx,
                         const DenseTensor& index,
                         const DenseTensor& out_grad,
                         int dim,
                         DenseTensor* x_grad) {
  if (out_grad.numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }

  dev_ctx.template Alloc<T>(x_grad);

  auto out_grad_dims = out_grad.dims();
  const int rank = out_grad_dims.size();

  if (dim < 0) {
    dim += rank;
  }

  PADDLE_ENFORCE_GE(
      dim,
      0,
      common::errors::InvalidArgument("The dimension index should be greater "
                                      "than or equal to 0, but got %d.",
                                      dim));
  PADDLE_ENFORCE_LT(
      dim,
      rank,
      common::errors::InvalidArgument(
          "The dimension index should be less than rank %d, but got %d.",
          rank,
          dim));

  if (index.numel() == 0) {
    Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    return;
  }

  LaunchIndexFillGradCudaKernel<T, Context>(
      dev_ctx, index, out_grad, dim, x_grad);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(index_fill,
                          musa,
                          ALL_LAYOUT,
                          phi::IndexFillKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool,
                          int16_t,
                          uint8_t,
                          int8_t,
                          phi::float16,
                          phi::bfloat16,
                          phi::complex64,
                          phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(index_fill_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::IndexFillGradKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          bool,
                          int16_t,
                          uint8_t,
                          int8_t,
                          phi::float16,
                          phi::bfloat16,
                          phi::complex64,
                          phi::complex128) {}
