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

#pragma once
#include <type_traits>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"

#include "paddle/phi/core/memory/allocation/allocator_facade.h"
#include "paddle/phi/kernels/transpose_kernel.h" 
#include "paddle/phi/kernels/gpu/flash_attn_utils.h"

#include "runtime/utils.h"

using namespace phi;
using ::musa::dnn::ScaledDotProductAttention;
using ::musa::dnn::MemoryHandler;
using muTensor = ::musa::dnn::Tensor;
using vec_128bit  = uint32_t __attribute__((vector_size(16)));

namespace phi {

template <typename T>
inline constexpr bool IsBf16() {
  return std::is_same<T, phi::bfloat16>::value;
}

}  // namespace phi

#define CHECK_MUDNN_STATUS(rst, msg)                                        \
PADDLE_ENFORCE_EQ(                                                          \
    rst,                                                                    \
    ::musa::dnn::Status::SUCCESS,                                           \
    phi::errors::External(                                                  \
        "%s MUDNN failed in: %s",                                           \
        __FUNCTION__,                                                       \
        msg))

template <typename T>
__global__ void build_padding_mask_kernel(const int* cu_q,
                                         const int* cu_k,
                                         T* mask,
                                         int B,
                                         int max_q,
                                         int max_k,
                                         float negInf) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;  // key position
  int i = blockIdx.y * blockDim.y + threadIdx.y;  // query position
  int b = blockIdx.z;

  if (b >= B || i >= max_q || j >= max_k) return;

  int q_len = cu_q[b + 1] - cu_q[b];
  int k_len = cu_k[b + 1] - cu_k[b];

  bool q_valid = i < q_len;
  bool k_valid = j < k_len;

  int64_t offset = ((int64_t)b * max_q + i) * (int64_t)max_k + j;

  mask[offset] = (q_valid && k_valid) ? static_cast<T>(0) : static_cast<T>(negInf);
}

template <typename T, int BLOCKDIM_X, int BLOCKDIM_Y, bool IsPaddingKV>
__global__ void padding_kernel(
    const T* input1,        
    const T* input2,        
    T* output1,             
    T* output2,
    const int* accum_length, 
    int max_seq,            
    int head_dim,           
    int num_heads,          
    int batch_size,
    int seq_stride,int head_stride,int seq_out_stride,int head_out_stride,int bs_out_stride
) {
    const int input_base_idx = blockIdx.y * head_stride + (accum_length[blockIdx.z] + blockIdx.x * BLOCKDIM_Y + threadIdx.y) * seq_stride;
    const int out_base_idx = blockIdx.z * bs_out_stride + blockIdx.y*head_out_stride;

    const int seq_length = accum_length[blockIdx.z+1] - accum_length[blockIdx.z];
    
    const int seq_out_idx = blockIdx.x * BLOCKDIM_Y + threadIdx.y;

    bool is_oob = threadIdx.x * 16 / sizeof(T) >= head_dim;

    if (seq_out_idx >= max_seq || is_oob) {
        return;
    }

    int output_pos = out_base_idx + seq_out_idx * seq_out_stride;
    const vec_128bit* input_vec1  = reinterpret_cast<const vec_128bit*>(input1 + input_base_idx);
    const vec_128bit* input_vec2  = reinterpret_cast<const vec_128bit*>(input2 + input_base_idx);

    vec_128bit* output_vec_1 = reinterpret_cast<vec_128bit*>(output1 + output_pos);
    vec_128bit* output_vec_2 = reinterpret_cast<vec_128bit*>(output2 + output_pos);
 
    if (seq_out_idx < seq_length) {
        output_vec_1[threadIdx.x] = input_vec1[threadIdx.x];
        if constexpr (IsPaddingKV) {
            output_vec_2[threadIdx.x] = input_vec2[threadIdx.x];
        }
    } else {
        vec_128bit Zeros = {};
        output_vec_1[threadIdx.x] =  Zeros;
        if constexpr (IsPaddingKV) {
            output_vec_2[threadIdx.x] =  Zeros;
        }
    }
}

template <typename T, int BLOCKDIM_X, int BLOCKDIM_Y>
__global__ void unpadding_kernel(
    const T* input,         //  [max_seq, head_dim, num_heads, batch_size]
    T* output,              //  [sum_seq, head_dim, num_heads]
    const int* accum_length,//  [seq_0, seq_1,...]
    int sum_seq,            // (seq_lens)）
    int max_seq,            
    int head_dim,           
    int num_heads,          
    int batch_size,
    int seq_stride,int head_stride,int bs_stride, int seq_out_stride,int head_out_stride
) {
    const int out_base_idx = blockIdx.y * head_out_stride + (accum_length[blockIdx.z] + blockIdx.x * BLOCKDIM_Y + threadIdx.y) * seq_out_stride;
    const int in_base_idx = blockIdx.z * bs_stride + blockIdx.y*head_stride;

    const int seq_length = accum_length[blockIdx.z+1] - accum_length[blockIdx.z];
    
    const int seq_in_idx = blockIdx.x * BLOCKDIM_Y + threadIdx.y;
    bool is_oob = threadIdx.x * 16 / sizeof(T) >= head_dim;

    if (seq_in_idx >= max_seq || is_oob) {
        return;
    }

    int input_pos = in_base_idx + seq_in_idx * seq_stride;
    vec_128bit* output_vec  = reinterpret_cast<vec_128bit*>(output + out_base_idx);

    const vec_128bit* input_vec = reinterpret_cast<const vec_128bit*>(input + input_pos);

    if (seq_in_idx < seq_length) {
        output_vec[threadIdx.x] = input_vec[threadIdx.x];
    }
}

template<class T, typename Context>
inline void launch_unpadding_kernel(
    const Context& dev_ctx,
    const T* input,
    T* output,
    const int* accum_lens,
    int sum_seq,
    int max_seq,
    int head_dim,
    int num_heads,
    int batch_size,
    int seq_out_stride, int head_out_stride,int seq_stride,int head_stride,int bs_stride
) {
    auto stream = dev_ctx.stream();
    if (head_dim > 128) {

        dim3 block(32, 32);  
        dim3 grid(
            (max_seq + block.y - 1) / block.y,
            num_heads,
            batch_size
        );
        unpadding_kernel<T, 32, 32><<<grid, block, 0, stream>>>(
        input, output,accum_lens,
        sum_seq, max_seq, head_dim, num_heads, batch_size, 
        seq_stride, head_stride,bs_stride, seq_out_stride, head_out_stride
        );
    } 
    else if (head_dim > 64) {
        dim3 block(16, 32);  
        dim3 grid(
            (max_seq + block.y - 1) / block.y,
            num_heads,
            batch_size
        );
        unpadding_kernel<T, 16, 32><<<grid, block, 0, stream>>>(
        input, output, accum_lens,
        sum_seq, max_seq, head_dim, num_heads, batch_size,
        seq_stride, head_stride,bs_stride, seq_out_stride, head_out_stride
        );
    } else if (head_dim > 32) {
        dim3 block(8, 64);  
        dim3 grid(
            (max_seq + block.y - 1) / block.y,
            num_heads,
            batch_size
        );
        unpadding_kernel<T, 8, 64><<<grid, block, 0, stream>>>(
        input, output,accum_lens,
        sum_seq, max_seq, head_dim, num_heads, batch_size,
        seq_stride, head_stride,bs_stride, seq_out_stride, head_out_stride
        );
    } else {
        RaiseNotSupportedError();
    }
}

template <class T, typename Context>
inline void launch_padding_kernel(
    const Context& dev_ctx,
    const T* input_q,
    const T* input_k,
    const T* input_v,
    const T* input_dout,
    const T* input_out,
    T* output_q,
    T* output_k,
    T* output_v,
    T* output_dout,
    T* output_out,
    const int* accum_lens_q,
    const int* accum_lens_k,
    int seqlen_q,
    int seqlen_kv,
    int head_dim_q,
    int head_dim_k,
    int head_dim_v,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    int batch_size,
    int q_seq_stride, int q_head_stride,int q_seq_out_stride,int q_head_out_stride,int q_bs_out_stride,
    int k_seq_stride, int k_head_stride,int k_seq_out_stride, int k_head_out_stride,int k_bs_out_stride,
    int v_seq_stride, int v_head_stride,int v_seq_out_stride, int v_head_out_stride,int v_bs_out_stride
) {
  auto stream = dev_ctx.stream();
  if (head_dim_q > 128) {
        dim3 block(32, 32);  
        dim3 grid_q(
            (seqlen_q + block.y - 1) / block.y,
            num_heads_q,
            batch_size
        );
        dim3 grid_k(
            (seqlen_kv + block.y - 1) / block.y,
            num_heads_k,
            batch_size
        );
        dim3 grid_v(
            (seqlen_kv + block.y - 1) / block.y,
            num_heads_v,
            batch_size
        );
        padding_kernel<T, 32, 32, false><<<grid_q, block, 0, stream>>>(
        input_q, nullptr, output_q, nullptr, accum_lens_q,
        seqlen_q, head_dim_q, num_heads_q, batch_size,
        q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride
        );
        if (input_out) {
          padding_kernel<T, 32, 32, false><<<grid_q, block, 0, stream>>>(
          input_out, nullptr, output_out, nullptr, accum_lens_q,
          seqlen_q, head_dim_q, num_heads_q, batch_size,
          q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride
          );
          padding_kernel<T, 32, 32, false><<<grid_q, block, 0, stream>>>(
          input_dout, nullptr, output_dout, nullptr, accum_lens_q,
          seqlen_q, head_dim_q, num_heads_q, batch_size,
          q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride
          );
        }
        padding_kernel<T, 32, 32, false><<<grid_k, block, 0, stream>>>(
        input_k, nullptr, output_k, nullptr,accum_lens_k,
        seqlen_kv, head_dim_k, num_heads_k, batch_size, 
        k_seq_stride, k_head_stride,k_seq_out_stride, k_head_out_stride,k_bs_out_stride
        );
        padding_kernel<T, 32, 32, false><<<grid_v, block, 0, stream>>>(
        input_v, nullptr, output_v, nullptr,accum_lens_k,
        seqlen_kv, head_dim_v, num_heads_v, batch_size, 
        v_seq_stride, v_head_stride,v_seq_out_stride, v_head_out_stride,v_bs_out_stride
        );

    } 
    else if (head_dim_q > 64) {
        dim3 block(16, 32);  
        dim3 grid_q(
            (seqlen_q+ block.y - 1) / block.y,
            num_heads_q,
            batch_size
        );
        dim3 grid_k(
            (seqlen_kv + block.y - 1) / block.y,
            num_heads_k,
            batch_size
        );
        dim3 grid_v(
            (seqlen_kv + block.y - 1) / block.y,
            num_heads_v,
            batch_size
        );
        padding_kernel<T, 16, 32, false><<<grid_q, block, 0, stream>>>(
        input_q, nullptr, output_q, nullptr, accum_lens_q,
        seqlen_q, head_dim_q, num_heads_q, batch_size,
        q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride
        );
        if (input_out) {
          padding_kernel<T, 16, 32, false><<<grid_q, block, 0, stream>>>(
          input_out, nullptr, output_out, nullptr, accum_lens_q,
          seqlen_q, head_dim_q, num_heads_q, batch_size,
          q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride
          );
          padding_kernel<T, 16, 32, false><<<grid_q, block, 0, stream>>>(
          input_dout, nullptr, output_dout, nullptr, accum_lens_q,
          seqlen_q, head_dim_q, num_heads_q, batch_size,
          q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride
          );
        }
        padding_kernel<T, 16, 32, false><<<grid_k, block, 0, stream>>>(
        input_k, nullptr, output_k, nullptr,accum_lens_k,
        seqlen_kv, head_dim_k, num_heads_k, batch_size,
        k_seq_stride, k_head_stride,k_seq_out_stride, k_head_out_stride,k_bs_out_stride
        );
        padding_kernel<T, 16, 32, false><<<grid_v, block, 0, stream>>>(
        input_v, nullptr, output_v, nullptr,accum_lens_k,
        seqlen_kv, head_dim_v, num_heads_v, batch_size,
        v_seq_stride, v_head_stride,v_seq_out_stride, v_head_out_stride,v_bs_out_stride
        );
    } else if (head_dim_q > 32) {
        dim3 block(8, 64);  
        dim3 grid_q(
            (seqlen_q + block.y - 1) / block.y,
            num_heads_q,
            batch_size
        );
        dim3 grid_k(
            (seqlen_kv + block.y - 1) / block.y,
            num_heads_k,
            batch_size
        );
        dim3 grid_v(
            (seqlen_kv +  block.y - 1) / block.y,
            num_heads_v,
            batch_size
        );
        padding_kernel<T, 8, 64, false><<<grid_q, block, 0, stream>>>(
        input_q, nullptr, output_q, nullptr, accum_lens_q,
        seqlen_q, head_dim_q, num_heads_q, batch_size,
        q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride
        );
        if (input_out) {
          padding_kernel<T, 8, 64, false><<<grid_q, block, 0, stream>>>(
          input_out, nullptr, output_out, nullptr, accum_lens_q,
          seqlen_q, head_dim_q, num_heads_q, batch_size,
          q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride
          );
          padding_kernel<T, 8, 64, false><<<grid_q, block, 0, stream>>>(
          input_dout, nullptr, output_dout, nullptr, accum_lens_q,
          seqlen_q, head_dim_q, num_heads_q, batch_size,
          q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride
          );
        }
        padding_kernel<T, 8, 64, false><<<grid_k, block, 0, stream>>>(
        input_k, nullptr, output_k, nullptr,accum_lens_k,
        seqlen_kv, head_dim_k, num_heads_k, batch_size,
        k_seq_stride, k_head_stride,k_seq_out_stride, k_head_out_stride,k_bs_out_stride
        );
        padding_kernel<T, 8, 64, false><<<grid_v, block, 0, stream>>>(
        input_v, nullptr, output_v, nullptr,accum_lens_k,
        seqlen_kv, head_dim_v, num_heads_v, batch_size,
        v_seq_stride, v_head_stride,v_seq_out_stride, v_head_out_stride,v_bs_out_stride
        );
    } else {
        RaiseNotSupportedError();
    }
}

template <typename T, typename Context>
inline void VarlenFaSeqlenUnpad(
    const Context& dev_ctx,
    DenseTensor &in,  
    DenseTensor* out,  
    const DenseTensor &accum_lens,
    int sum_seq,
    int max_seq,
    int head_dim,
    int num_heads,
    int batch_size) { 

    int seq_out_stride = out->strides()[0];
    int head_out_stride = out->strides()[1];

    int seq_stride = in.strides()[2];
    int head_stride = in.strides()[1];
    int bs_stride = in.strides()[0];

    auto stream = dev_ctx.stream();

    if (in.dtype() == phi::DataType::BFLOAT16) 
    {
        launch_unpadding_kernel<__mt_bfloat16, Context>(
            dev_ctx,
            reinterpret_cast<const __mt_bfloat16*>(in.data<T>()),
            reinterpret_cast<__mt_bfloat16*>(out->data<T>()),
            accum_lens.data<int>(), 
            sum_seq, 
            max_seq, 
            head_dim, 
            num_heads,
            batch_size,
            seq_out_stride,head_out_stride,seq_stride,head_stride,bs_stride);
    }
    else if (in.dtype() == phi::DataType::FLOAT16)
    {
        launch_unpadding_kernel<__half, Context>(
            dev_ctx,
            reinterpret_cast<const __half*>(in.data<T>()),
            reinterpret_cast<__half*>(out->data<T>()),
            accum_lens.data<int>(), 
            sum_seq, 
            max_seq, 
            head_dim,
            num_heads,
            batch_size,
            seq_out_stride,head_out_stride,seq_stride,head_stride,bs_stride);
    }
    else if (in.dtype() == phi::DataType::FLOAT32)
    {
        launch_unpadding_kernel<float, Context>(
            dev_ctx,
            reinterpret_cast<const float*>(in.data<T>()),
            reinterpret_cast<float*>(out->data<T>()),
            accum_lens.data<int>(), 
            sum_seq, 
            max_seq, 
            head_dim,
            num_heads,
            batch_size,
            seq_out_stride,head_out_stride,seq_stride,head_stride,bs_stride);
    }
    else
    {
        RaiseNotSupportedError();
    }
}

template <typename T, typename Context>
inline void VarlenFaSeqlenPad(
    const Context& dev_ctx,
    const DenseTensor& q_in,
    const DenseTensor& k_in,
    const DenseTensor& v_in,
    const paddle::optional<DenseTensor>& dout_in,
    const paddle::optional<DenseTensor>& out_in,
    DenseTensor& q_out,
    DenseTensor& k_out,
    DenseTensor& v_out,
    paddle::optional<DenseTensor>& dout_out,
    paddle::optional<DenseTensor>& out_out,
    const DenseTensor& cu_seqlens_q,
    const DenseTensor& cu_seqlens_k,
    int64_t sum_seq,
    int64_t max_seq,
    int64_t batch_size) {
  int q_seq_stride = q_in.strides()[0];
  int q_head_stride = q_in.strides()[1];

  int q_seq_out_stride = q_out.strides()[2];
  int q_head_out_stride = q_out.strides()[1];
  int q_bs_out_stride = q_out.strides()[0];

  int k_seq_stride = k_in.strides()[0];
  int k_head_stride = k_in.strides()[1];

  int k_seq_out_stride = k_out.strides()[2];
  int k_head_out_stride = k_out.strides()[1];
  int k_bs_out_stride = k_out.strides()[0];

  int v_seq_stride = v_in.strides()[0];
  int v_head_stride = v_in.strides()[1];

  int v_seq_out_stride = v_out.strides()[2];
  int v_head_out_stride = v_out.strides()[1];
  int v_bs_out_stride = v_out.strides()[0];
  
  int head_dim_q = q_in.dims()[2];
  int head_dim_k = k_in.dims()[2];
  int head_dim_v = v_in.dims()[2];

  int num_heads_q = q_in.dims()[1];
  int num_heads_k = k_in.dims()[1];
  int num_heads_v = v_in.dims()[1];

  int seqlen_q = q_out.dims()[2];
  int seqlen_kv = k_out.dims()[2];

  // const at::musa::OptionalMUSAGuard device_guard(device_of(q_in));
  auto stream = dev_ctx.stream();

  T* out_in_ptr = nullptr;
  T* out_out_ptr = nullptr;
  T* dout_in_ptr = nullptr;
  T* dout_out_ptr = nullptr;

  if (dout_in.is_initialized()) {
    auto dout_in_tmp = dout_in;
    dout_in_ptr = dout_in_tmp.get().data<T>();
    dout_out_ptr = dout_out.get().data<T>();

    auto out_in_tmp = out_in;
    out_in_ptr = out_in_tmp.get().data<T>();
    out_out_ptr = out_out.get().data<T>();
  }

  if (q_in.dtype() == phi::DataType::BFLOAT16) {
    launch_padding_kernel<__mt_bfloat16, Context>(
            dev_ctx,
            reinterpret_cast<const __mt_bfloat16*>(q_in.data<T>()),
            reinterpret_cast<const __mt_bfloat16*>(k_in.data<T>()),
            reinterpret_cast<const __mt_bfloat16*>(v_in.data<T>()),
            reinterpret_cast<const __mt_bfloat16*>(dout_in_ptr),
            reinterpret_cast<const __mt_bfloat16*>(out_in_ptr),
            reinterpret_cast<__mt_bfloat16*>(q_out.data<T>()),
            reinterpret_cast<__mt_bfloat16*>(k_out.data<T>()),
            reinterpret_cast<__mt_bfloat16*>(v_out.data<T>()),
            reinterpret_cast<__mt_bfloat16*>(dout_out_ptr),
            reinterpret_cast<__mt_bfloat16*>(out_out_ptr),
            cu_seqlens_q.data<int>(), 
            cu_seqlens_k.data<int>(), 
            seqlen_q, 
            seqlen_kv, 
            head_dim_q, 
            head_dim_k,
            head_dim_v,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            batch_size,
            q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride,
            k_seq_stride, k_head_stride,k_seq_out_stride,k_head_out_stride, k_bs_out_stride,
            v_seq_stride, v_head_stride,v_seq_out_stride,v_head_out_stride, v_bs_out_stride);
  } else if (q_in.dtype() == phi::DataType::FLOAT16) {
    launch_padding_kernel<__half, Context>(
            dev_ctx,
            reinterpret_cast<const __half*>(q_in.data<T>()),
            reinterpret_cast<const __half*>(k_in.data<T>()),
            reinterpret_cast<const __half*>(v_in.data<T>()),
            reinterpret_cast<const __half*>(dout_in_ptr),
            reinterpret_cast<const __half*>(out_in_ptr),
            reinterpret_cast<__half*>(q_out.data<T>()),
            reinterpret_cast<__half*>(k_out.data<T>()),
            reinterpret_cast<__half*>(v_out.data<T>()),
            reinterpret_cast<__half*>(dout_out_ptr),
            reinterpret_cast<__half*>(out_out_ptr),
            cu_seqlens_q.data<int>(), 
            cu_seqlens_k.data<int>(), 
            seqlen_q, 
            seqlen_kv, 
            head_dim_q, 
            head_dim_k,
            head_dim_v,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            batch_size,
            q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride,
            k_seq_stride, k_head_stride,k_seq_out_stride,k_head_out_stride, k_bs_out_stride,
            v_seq_stride, v_head_stride,v_seq_out_stride,v_head_out_stride, v_bs_out_stride);
  } else if (q_in.dtype() == phi::DataType::FLOAT32) {
    launch_padding_kernel<float, Context>(
            dev_ctx,
            reinterpret_cast<const float*>(q_in.data<T>()),
            reinterpret_cast<const float*>(k_in.data<T>()),
            reinterpret_cast<const float*>(v_in.data<T>()),
            reinterpret_cast<const float*>(dout_in_ptr),
            reinterpret_cast<const float*>(out_in_ptr),
            reinterpret_cast<float*>(q_out.data<T>()),
            reinterpret_cast<float*>(k_out.data<T>()),
            reinterpret_cast<float*>(v_out.data<T>()),
            reinterpret_cast<float*>(dout_out_ptr),
            reinterpret_cast<float*>(out_out_ptr),
            cu_seqlens_q.data<int>(), 
            cu_seqlens_k.data<int>(), 
            seqlen_q, 
            seqlen_kv, 
            head_dim_q, 
            head_dim_k,
            head_dim_v,
            num_heads_q,
            num_heads_k,
            num_heads_v,
            batch_size,
            q_seq_stride,q_head_stride,q_seq_out_stride,q_head_out_stride,q_bs_out_stride,
            k_seq_stride, k_head_stride,k_seq_out_stride,k_head_out_stride, k_bs_out_stride,
            v_seq_stride, v_head_stride,v_seq_out_stride,v_head_out_stride, v_bs_out_stride);
  } else {
    RaiseNotSupportedError();
  }
}

inline void DebugCheckTensorMemory(const phi::DenseTensor& t,
                                   const std::string& name) {
  try {
    switch (t.dtype()) {
      case phi::DataType::FLOAT16:
        (void)t.data<phi::dtype::float16>();
        break;
      case phi::DataType::FLOAT32:
        (void)t.data<float>();
        break;
      case phi::DataType::BFLOAT16:
        (void)t.data<phi::dtype::bfloat16>();
        break;
      default:
        break;
    }
  } catch (const std::exception& e) {
    PADDLE_THROW(phi::errors::PreconditionNotMet(
        "[FlashAttn Debug] Tensor %s has mismatched dims/memory. "
        "Original error: %s",
        name.c_str(),
        e.what()));
  }
}

template <typename T, typename Context>
inline void TransposeMudnnStyle(const Context& dev_ctx,
                          const phi::DenseTensor& in,
                          phi::DenseTensor* out) {
  const auto& d = in.dims();
  PADDLE_ENFORCE_EQ(
    in.dims().size(), 4,
    phi::errors::InvalidArgument(
      "TransposeMudnnStyle expects 4D tensor, but got %dD.", in.dims().size()));
  phi::DenseTensorMeta meta(
      in.dtype(),
      phi::make_ddim({d[0], d[2], d[1], d[3]}),
      in.layout());
  out->set_meta(meta);
  std::vector<int> axis = {0, 2, 1, 3};
  phi::TransposeKernel<T, Context>(dev_ctx, in, axis, out);
}
 
template <typename Handle_t, void Create(Handle_t*), void Destroy(Handle_t)>
struct DeviceThreadHandlePool
    : public std::enable_shared_from_this<
          DeviceThreadHandlePool<Handle_t, Create, Destroy>> {
  struct Handle {
    Handle_t handle;
    Handle(bool create = false) : handle(nullptr) {
      if (create)
        Create(&handle);
    }

    Handle(const Handle& rhs) = delete;
    Handle(Handle&& rhs) : Handle() {
      std::swap(handle, rhs.handle);
    }
    Handle& operator=(Handle rhs) {
      std::swap(handle, rhs.handle);
      return *this;
    }
    ~Handle() {
      if (handle)
        Destroy(handle);
    }
  };

  std::mutex mutex;
  std::unordered_map<int, std::vector<Handle>> created_handles;
  std::unordered_map<int, std::vector<Handle_t>> available_handles;

  class PoolWindow {
   public:
    PoolWindow(std::shared_ptr<DeviceThreadHandlePool> parent)
        : weak_parent_(std::move(parent)) {}
    ~PoolWindow() {
      release();
    }

    Handle_t reserve(int device) {
      if (internal_handles_.find(device) != internal_handles_.end())
        return internal_handles_[device];

      auto parent = weak_parent_.lock();
      PADDLE_ENFORCE_NE(parent, nullptr,
                  phi::errors::Fatal("Cannot create handle during program termination"));
      std::lock_guard<std::mutex> guard(parent->mutex);

      if (parent->available_handles[device].size() > 0) {
        internal_handles_[device] = parent->available_handles[device].back();
        parent->available_handles[device].pop_back();
      } else {
        parent->created_handles[device].emplace_back(true /*create*/);
        internal_handles_[device] =
            parent->created_handles[device].back().handle;
      }

      return internal_handles_[device];
    }

   private:
    std::unordered_map<int, Handle_t> internal_handles_;

    std::weak_ptr<DeviceThreadHandlePool> weak_parent_;

    void release() {
      if (internal_handles_.size() > 0) {
        auto parent = weak_parent_.lock();
        if (!parent) {
          return;
        }

        std::lock_guard<std::mutex> guard(parent->mutex);
        for (auto d_h : internal_handles_)
          parent->available_handles[d_h.first].push_back(d_h.second);
      }
    }
  };

  PoolWindow* newPoolWindow() {
    return new PoolWindow(this->shared_from_this());
  }

  std::unique_ptr<PoolWindow> NewPoolWindow() {
    return std::make_unique<PoolWindow>(this->shared_from_this());
  }
};

inline void CreateMuDNNHandle(mudnnHandle_t* handle) {
  PADDLE_ENFORCE_NOT_NULL(
    handle,
    phi::errors::Fatal("Handle pointer is no-nullptr"));
  int device;
  musaGetDevice(&device);
  PADDLE_ENFORCE_GE(device, -1, phi::errors::Fatal("Device must be greater or equal to 0"));
  *handle = new ::musa::dnn::Handle(device);
}

inline void DestroyMuDNNHandle(mudnnHandle_t /*handle*/) {}

using MudnnPoolType = DeviceThreadHandlePool<
    mudnnHandle_t,
    CreateMuDNNHandle,
    DestroyMuDNNHandle>;

inline MemoryHandler PaddleInternalMemAlloc(size_t bytes,
                                            const phi::Place& place) {
  using paddle::memory::allocation::AllocatorFacade;
  using paddle::memory::allocation::AllocationPtr;

  if (bytes == 0) {
    return MemoryHandler(nullptr, [](void*) {});
  }

  auto allocator = AllocatorFacade::Instance().GetAllocator(place);
  AllocationPtr alloc = allocator->Allocate(bytes);
  void* ptr = alloc->ptr();

  auto holder = std::make_shared<AllocationPtr>(std::move(alloc));

  return MemoryHandler(
      ptr,
      [holder](void*) mutable {});
}

inline void SetMUTensorDType(phi::DataType dtype, muTensor& m_t) {
  using Type = muTensor::Type;

  switch (dtype) {
    case phi::DataType::FLOAT16:
      m_t.SetType(Type::HALF);
      break;
    case phi::DataType::FLOAT32:
      m_t.SetType(Type::FLOAT);
      break;
    case phi::DataType::FLOAT64:
      m_t.SetType(Type::DOUBLE);
      break;
    case phi::DataType::INT16:
      m_t.SetType(Type::INT16);
      break;
    case phi::DataType::INT32:
      m_t.SetType(Type::INT32);
      break;
    case phi::DataType::INT64:
      m_t.SetType(Type::INT64);
      break;
    case phi::DataType::INT8:
      m_t.SetType(Type::INT8);
      break;
    case phi::DataType::UINT8:
      m_t.SetType(Type::UINT8);
      break;
    case phi::DataType::UINT64:
      m_t.SetType(Type::UINT64);
      break;
    case phi::DataType::BOOL:
      m_t.SetType(Type::BOOL);
      break;
    case phi::DataType::BFLOAT16:
      m_t.SetType(Type::BFLOAT16);
      break;
#if defined(REAL_MUSA_VERSION) && REAL_MUSA_VERSION >= 4000
    case phi::DataType::FLOAT8_E5M2:
      m_t.SetType(muTensor::Type::FP8_E5M2);
      break;
    case phi::DataType::::FLOAT8_E4M3FN:
      m_t.SetType(muTensor::Type::FP8_E4M3);
      break;
#endif
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "SetMUTensorDType: unsupported paddle dtype %s",
          phi::DataTypeToString(dtype)));
  }
}

inline void SetMUTensorAddr(const void* addr, muTensor& m_t) {
  m_t.SetAddr(addr);
}

inline void ConfigFormat(
    const phi::DenseTensor& t,
    muTensor& mt,
    bool permute_if_not_contiguous) {
  const auto dims = t.dims();
  const int ndim = dims.size();

  PADDLE_ENFORCE_LE(
      ndim,
      8,
      phi::errors::InvalidArgument(
          "mudnn only supports input tensors with dim <= 8, but got %d.",
          ndim));

  const auto& meta = t.meta();

  if (!meta.is_contiguous() && permute_if_not_contiguous) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Non-contiguous tensor is not supported in Paddle-MUSA ConfigFormat. "
        "dims = %s, layout = %d, permute_if_not_contiguous = %d. "
        "Please make the tensor contiguous before calling ConfigFormat.",
        dims,
        static_cast<int>(meta.layout),
        static_cast<int>(permute_if_not_contiguous)));
  }

  muTensor::Format mudnn_format = muTensor::Format::NCHW;
  auto layout = meta.layout;

  if (ndim == 4) {
    if (layout == phi::DataLayout::NHWC) {
      mudnn_format = muTensor::Format::NHWC;
    } else {
      mudnn_format = muTensor::Format::NCHW;
    }
  } else if (ndim == 5) {
    if (layout == phi::DataLayout::NDHWC) {
      mudnn_format = muTensor::Format::NDHWC;
    } else {
      mudnn_format = muTensor::Format::NCDHW;
    }
  } else {
    mudnn_format = muTensor::Format::NCHW;
  }

  mt.SetFormat(mudnn_format);

  std::vector<int64_t> sizes(ndim);
  for (int i = 0; i < ndim; ++i) {
    sizes[i] = static_cast<int64_t>(dims[i]);
  }

  std::vector<int64_t> strides(ndim);
  int64_t stride = 1;
  for (int i = ndim - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= sizes[i];
  }

  mt.SetNdInfo(ndim, sizes.data(), strides.data());
}

inline muTensor CreateMUTensor(const DenseTensor& t,
                               bool permute_if_not_contiguous = false) {
  if (!t.initialized()) {
    return muTensor();
  }
  muTensor rst;
  SetMUTensorDType(t.dtype(), rst);
  const void* addr = t.data();
  SetMUTensorAddr(addr, rst);
  ConfigFormat(t, rst, permute_if_not_contiguous);

  return rst;
}

enum class MaskType {
  NONE = 0,
  CAUSAL = 1,
  // Assume that full mask shape is [N, H, L, S]
  // Floating masks
  FLT_LS         = 100, // 2D
  FLT_NS         = 101, // key padding
  FLT_HLS_BCAST  = 102, // 3-dim no batch broadcast
  FLT_HLS_FULL   = 103, // 3-dim no batch no broadcast
  FLT_NHLS_BCAST = 104, // 4-dim broadcast
  FLT_NHLS_FULL  = 105, // no broadcast
  // Binary masks
  BIN_LS         = 200, // 2D
  BIN_NS         = 201, // key padding
  BIN_HLS_BCAST  = 202, // 3-dim no batch broadcast
  BIN_HLS_FULL   = 203, // 3-dim no batch no broadcast
  BIN_NHLS_BCAST = 204, // 4-dim broadcast
  BIN_NHLS_FULL  = 205, // no broadcast
};
// clang-format on

inline bool IsExpandableTo(const phi::DDim& from,
                           std::initializer_list<int64_t> to) {
  std::vector<int64_t> to_vec(to);
  int64_t ndim_from = from.size();
  int64_t ndim_to = static_cast<int64_t>(to_vec.size());

  for (int64_t i = 0; i < ndim_to; ++i) {
    int64_t to_dim = to_vec[ndim_to - 1 - i];
    int64_t from_dim = 1;
    if (ndim_from - 1 - i >= 0) {
      from_dim = from[ndim_from - 1 - i];
    }
    if (!(from_dim == to_dim || from_dim == 1 || to_dim == 1)) {
      return false;
    }
  }
  return true;
}

inline MaskType ParseMaskType(const paddle::optional<DenseTensor>& mask,
                       bool is_causal,
                       int64_t N,
                       int64_t H_q,
                       int64_t L,
                       int64_t S) {
  if (is_causal) {
    return MaskType::CAUSAL;
  }

  if (!mask || !mask->initialized()) {
    return MaskType::NONE;
  }

  const auto& m = mask.get();
  const auto& dims = m.dims();
  const int64_t ndim = dims.size();
  const bool is_binary_mask = (m.dtype() == phi::DataType::BOOL);

  if (ndim == 2) {
    const int64_t d0 = dims[0];
    const int64_t d1 = dims[1];

    if (d0 == L && d1 == S) {
      return is_binary_mask ? MaskType::BIN_LS : MaskType::FLT_LS;
    }
    if (d0 == N && d1 == S) {
      return is_binary_mask ? MaskType::BIN_NS : MaskType::FLT_NS;
    }

    PADDLE_THROW(phi::errors::InvalidArgument(
        "MUSA SPDA: shape of 2D attn_mask should be [L, S] or [N, S]."));
  }

  if (ndim == 3) {
    auto full = phi::make_ddim({H_q, L, S});
    if (dims == full) {
      return is_binary_mask ? MaskType::BIN_HLS_FULL
                            : MaskType::FLT_HLS_FULL;
    }

    PADDLE_ENFORCE_EQ(
        IsExpandableTo(dims, {H_q, L, S}),
        true,
        phi::errors::InvalidArgument(
            "MUSA SPDA: shape of 3D attn_mask should be expandable to "
            "[H_q, L, S]."));
    return is_binary_mask ? MaskType::BIN_HLS_BCAST
                          : MaskType::FLT_HLS_BCAST;
  }

  if (ndim == 4) {
    auto full = phi::make_ddim({N, H_q, L, S});
    if (dims == full) {
      return is_binary_mask ? MaskType::BIN_NHLS_FULL
                            : MaskType::FLT_NHLS_FULL;
    }

    PADDLE_ENFORCE_EQ(
        IsExpandableTo(dims, {N, H_q, L, S}),
        true,
        phi::errors::InvalidArgument(
            "MUSA SPDA: shape of 4D attn_mask should be expandable to "
            "[N, H_q, L, S]."));
    return is_binary_mask ? MaskType::BIN_NHLS_BCAST
                          : MaskType::FLT_NHLS_BCAST;
  }

  PADDLE_THROW(
      phi::errors::InvalidArgument("MUSA SPDA: attn_mask should be 2/3/4D "
                                   "Tensor."));
}

inline bool HasMask(MaskType m) noexcept {
  return (m != MaskType::NONE) && (m != MaskType::CAUSAL);
}

inline bool IsPadMask(MaskType m) noexcept {
  return (m == MaskType::FLT_NS) || (m == MaskType::BIN_NS);
}

template <typename Context>
inline ::musa::dnn::Handle& GetMudnnHandle(const Context& dev_ctx) {
  int device;
  musaGetDevice(&device);

  static auto pool = std::make_shared<MudnnPoolType>();
  thread_local std::unique_ptr<MudnnPoolType::PoolWindow> myPoolWindow(
      pool->NewPoolWindow());

  mudnnHandle_t handle = myPoolWindow->reserve(device);
  handle->SetStream(dev_ctx.stream());
  return *handle;
}
