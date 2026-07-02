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

#include "paddle/phi/kernels/gpu/partial_concat_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/partial_concat_funcs.h"

namespace phi {

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

template <class T>
__global__ void ConcatPartialMUSAKernel(T** in,
                                        T* out,
                                        int64_t all_length,
                                        int64_t in_batch_len,
                                        int64_t start_index,
                                        int64_t out_batch_len,
                                        int64_t part_length) {
  int64_t id =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  while (id < all_length) {
    int64_t bs_id = id / out_batch_len;
    int64_t bs_index = id % out_batch_len;
    int64_t var_id = bs_index / part_length;
    int64_t part_index = bs_index % part_length;
    int64_t in_id = start_index + part_index;
    const T* tmp = in[var_id];
    out[id] = tmp[bs_id * in_batch_len + in_id];
    id += blockDim.x * gridDim.x;
  }
}

template <class T>
__global__ void ConcatPartialGradMUSAKernel(T** in,
                                            const T* out,
                                            int64_t all_length,
                                            int64_t in_batch_len,
                                            int64_t start_index,
                                            int64_t out_batch_len,
                                            int64_t part_length) {
  int64_t id =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  while (id < all_length) {
    int64_t bs_id = id / out_batch_len;
    int64_t bs_index = id % out_batch_len;
    int64_t var_id = bs_index / part_length;
    int64_t part_index = bs_index % part_length;
    int64_t in_id = start_index + part_index;
    T* tmp = in[var_id];
    tmp[bs_id * in_batch_len + in_id] = out[id];
    id += blockDim.x * gridDim.x;
  }
}

template <typename T, typename Context>
void PartialConcatOpMUSAKernel(const Context& dev_ctx,
                               const std::vector<const DenseTensor*>& x,
                               int start_index,
                               int length,
                               DenseTensor* out) {
  auto in_vars = x;
  PADDLE_ENFORCE_EQ(in_vars[0] != nullptr,
                    true,
                    common::errors::InvalidArgument(
                        "The input of partial concat should not be null."));

  auto input_dim = in_vars[0]->dims();
  PADDLE_ENFORCE_EQ(input_dim.size(),
                    2,
                    common::errors::InvalidArgument(
                        "Only supports 2-D array with batch size in the 1st "
                        "dimension and data in the 2nd."));
  auto in_size = input_dim[1];
  start_index = ComputeStartIndex(start_index, in_size);

  auto partial_len = length;
  if (partial_len < 0) {
    partial_len = in_size - start_index;
  }

  int in_num = in_vars.size();
  int batch_size = input_dim[0];
  int out_batch_len = partial_len * in_num;
  int all_length = batch_size * out_batch_len;

  constexpr size_t theory_sm_threads = 1024;
  auto stream = dev_ctx.stream();
  auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  auto sm_count = max_threads / theory_sm_threads;
  size_t tile_size = 0;
  int grids;
  int blocks;
  auto ComputeKernelParameter = [&](size_t length) {
    if (length >= max_threads)
      tile_size = 1024;
    else if (length < max_threads && length > sm_count * 128)
      tile_size = 512;
    else if (length <= sm_count * 128)
      tile_size = 256;
    grids = CEIL_DIV(length, tile_size);
    blocks = tile_size;
  };

  T* out_data = dev_ctx.template Alloc<T>(out);

  std::vector<const T*> in_data;
  for (int i = 0; i < in_num; ++i) in_data.emplace_back(in_vars[i]->data<T>());

  auto tmp_in_array = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      in_data.size() * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  phi::memory_utils::Copy(dev_ctx.GetPlace(),
                          tmp_in_array->ptr(),
                          CPUPlace(),
                          reinterpret_cast<uint8_t*>(const_cast<T**>(in_data.data())),
                          in_data.size() * sizeof(T*),
                          dev_ctx.stream());

  T** in_array_data = reinterpret_cast<T**>(tmp_in_array->ptr());
  ComputeKernelParameter(all_length);
  ConcatPartialMUSAKernel<T><<<grids, blocks, 0, stream>>>(in_array_data,
                                                           out_data,
                                                           all_length,
                                                           in_size,
                                                           start_index,
                                                           out_batch_len,
                                                           partial_len);
}

template <typename T, typename Context>
void PartialConcatGradOpMUSAKernel(const Context& dev_ctx,
                                   const std::vector<const DenseTensor*>& x,
                                   const DenseTensor& out_grad,
                                   int start_index,
                                   int length,
                                   std::vector<DenseTensor*> x_grad) {
  auto ins = x;
  auto outs = x_grad;

  PADDLE_ENFORCE_EQ(ins[0] != nullptr,
                    true,
                    common::errors::InvalidArgument(
                        "The input of partial concat should not be null."));
  auto batch_size = ins[0]->dims()[0];
  auto in_size = ins[0]->dims()[1];
  start_index = ComputeStartIndex(start_index, in_size);
  auto partial_len = length;
  if (partial_len < 0) partial_len = in_size - start_index;

  auto in_num = ins.size();
  auto grad_batch_len = partial_len * in_num;
  auto all_length = grad_batch_len * batch_size;
  auto& place = *dev_ctx.eigen_device();
  for (size_t i = 0; i < outs.size(); ++i) {
    dev_ctx.template Alloc<T>(outs[i]);
    auto dxt = EigenVector<T>::Flatten(*outs[i]);
    dxt.device(place) = dxt.constant(static_cast<T>(0));
  }

  constexpr size_t theory_sm_threads = 1024;
  auto stream = dev_ctx.stream();
  auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
  auto sm_count = max_threads / theory_sm_threads;
  size_t tile_size = 0;
  int grids;
  int blocks;
  auto ComputeKernelParameter = [&](size_t length) {
    if (length >= max_threads)
      tile_size = 1024;
    else if (length < max_threads && length > sm_count * 128)
      tile_size = 512;
    else if (length <= sm_count * 128)
      tile_size = 256;
    grids = CEIL_DIV(length, tile_size);
    blocks = tile_size;
  };

  std::vector<const T*> out_data;
  for (size_t i = 0; i < in_num; ++i) {
    out_data.emplace_back(outs[i]->data<T>());
  }
  auto tmp_out_array = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      out_data.size() * sizeof(T*),
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));

  phi::memory_utils::Copy(dev_ctx.GetPlace(),
                          tmp_out_array->ptr(),
                          CPUPlace(),
                          reinterpret_cast<uint8_t*>(const_cast<T**>(out_data.data())),
                          out_data.size() * sizeof(T*),
                          dev_ctx.stream());

  T** out_grad_data = reinterpret_cast<T**>(tmp_out_array->ptr());
  ComputeKernelParameter(all_length);
  ConcatPartialGradMUSAKernel<T><<<grids, blocks, 0, stream>>>(out_grad_data,
                                                               out_grad.data<T>(),
                                                               all_length,
                                                               in_size,
                                                               start_index,
                                                               grad_batch_len,
                                                               partial_len);
}

#undef CEIL_DIV

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(partial_concat,
                          musa,
                          ALL_LAYOUT,
                          phi::PartialConcatOpMUSAKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::float16,
                          phi::complex64,
                          phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(partial_concat_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::PartialConcatGradOpMUSAKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::float16,
                          phi::complex64,
                          phi::complex128) {}
