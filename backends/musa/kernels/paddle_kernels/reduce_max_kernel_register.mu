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

#include <limits>
#include <type_traits>

#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/reduce.h"

namespace phi {

namespace {

constexpr int kLargeLastDimMaxBlockSize = 256;

template <typename T>
__global__ void LastDimMaxKernel(const T* x,
                                 T* out,
                                 int64_t outer_size,
                                 int64_t reduce_size) {
  int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= outer_size) {
    return;
  }

  const T* row_data = x + row * reduce_size;
  T thread_max = row_data[0];
  for (int64_t i = threadIdx.x; i < reduce_size; i += blockDim.x) {
    T val = row_data[i];
    if (val > thread_max) {
      thread_max = val;
    }
  }

  __shared__ T shared[kLargeLastDimMaxBlockSize];
  shared[threadIdx.x] = thread_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride && shared[threadIdx.x + stride] > shared[threadIdx.x]) {
      shared[threadIdx.x] = shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out[row] = shared[0];
  }
}

inline bool IsLargeLastDimReduce(const DenseTensor& x,
                                 const IntArray& dims,
                                 bool reduce_all) {
  if (reduce_all || x.dims().size() == 0) {
    return false;
  }

  const auto reduce_dims = dims.GetData();
  if (reduce_dims.size() != 1) {
    return false;
  }

  int64_t axis = reduce_dims[0];
  const int64_t rank = x.dims().size();
  if (axis < 0) {
    axis += rank;
  }
  if (axis != rank - 1) {
    return false;
  }

  return x.numel() * static_cast<int64_t>(sizeof(float)) >
         std::numeric_limits<int32_t>::max();
}

}  // namespace

template <typename T, typename Context>
PADDLE_API void MaxRawKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const IntArray& dims,
                             bool keep_dim,
                             bool reduce_all,
                             DenseTensor* out) {
  reduce_all = recompute_reduce_all(x, dims, reduce_all);
  if constexpr (std::is_same<T, float>::value) {
    if (IsLargeLastDimReduce(x, dims, reduce_all)) {
      const int64_t reduce_size = x.dims()[x.dims().size() - 1];
      const int64_t outer_size = x.numel() / reduce_size;
      T* out_data = dev_ctx.template Alloc<T>(out);
      LastDimMaxKernel<T><<<outer_size,
                            kLargeLastDimMaxBlockSize,
                            0,
                            dev_ctx.stream()>>>(
          x.data<T>(), out_data, outer_size, reduce_size);
      return;
    }
  }
  auto out_dtype = x.dtype();
  phi::Reduce<T, kps::MaxFunctor, kps::IdentityFunctor>(
      dev_ctx, x, reduce_all, dims.GetData(), keep_dim, out_dtype, out);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(max_raw,
                   musa,
                   ALL_LAYOUT,
                   phi::MaxRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::float8_e4m3fn,
                   phi::float8_e5m2) {}
