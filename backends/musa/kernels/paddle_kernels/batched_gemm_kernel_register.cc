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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/legacy/gpu/batched_gemm.h"

namespace phi {

template <typename T, typename Context>
void BatchedGEMMMusaKernel(const Context& dev_ctx,
                           const DenseTensor& lhs,
                           const DenseTensor& rhs,
                           const std::vector<int64_t>& batch_sizes,
                           bool trans_lhs,
                           bool trans_rhs,
                           DenseTensor* output) {
  dev_ctx.template Alloc<T>(output);
  if (output->numel() == 0) {
    return;
  }

  const auto lhs_dims = lhs.dims();
  const auto rhs_dims = rhs.dims();
  const T* lhs_data = lhs.data<T>();
  const T* rhs_data = rhs.data<T>();
  T* out_data = output->data<T>();
  auto blas = funcs::GetBlas<Context, T>(dev_ctx);

  if (!trans_lhs) {
    const int64_t hidden_in = lhs_dims[1];
    const int64_t hidden_out = trans_rhs ? rhs_dims[1] : rhs_dims[2];
    const int64_t rhs_stride = rhs_dims[1] * rhs_dims[2];
    int64_t lhs_offset = 0;
    int64_t out_offset = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(batch_sizes.size()); ++i) {
      const int64_t expert_bs = batch_sizes[i];
      blas.GEMM(CblasNoTrans,
                trans_rhs ? CblasTrans : CblasNoTrans,
                expert_bs,
                hidden_out,
                hidden_in,
                static_cast<T>(1),
                lhs_data + lhs_offset,
                rhs_data + i * rhs_stride,
                static_cast<T>(0),
                out_data + out_offset);
      lhs_offset += expert_bs * hidden_in;
      out_offset += expert_bs * hidden_out;
    }
  } else {
    const int64_t hidden_in = lhs_dims[1];
    const int64_t hidden_out = rhs_dims[1];
    int64_t lhs_offset = 0;
    int64_t rhs_offset = 0;
    int64_t out_offset = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(batch_sizes.size()); ++i) {
      const int64_t expert_bs = batch_sizes[i];
      blas.GEMM(CblasTrans,
                CblasNoTrans,
                hidden_in,
                hidden_out,
                expert_bs,
                static_cast<T>(1),
                lhs_data + lhs_offset,
                rhs_data + rhs_offset,
                static_cast<T>(0),
                out_data + out_offset);
      lhs_offset += expert_bs * hidden_in;
      rhs_offset += expert_bs * hidden_out;
      out_offset += hidden_in * hidden_out;
    }
  }
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(batched_gemm,
                          musa,
                          ALL_LAYOUT,
                          phi::BatchedGEMMMusaKernel,
                          float,
                          double,
                          phi::bfloat16) {}
