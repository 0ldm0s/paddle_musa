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

#ifdef STREAM_TYPE
#undef STREAM_TYPE
#define STREAM_TYPE void*
#endif

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/common/reduce_type.h"
#include "paddle/phi/core/distributed/utils.h"
#include "paddle/phi/core/distributed/xccl_comm_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/global_scatter_kernel.h"
#include "paddle/phi/kernels/gpu/partial_allgather_kernel.h"
#include "paddle/phi/kernels/mp_allreduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void GlobalScatterKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& local_count,
                         const DenseTensor& global_count,
                         DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      local_count.dtype(),
      DataType::INT64,
      common::errors::InvalidArgument("Please use int64 type in local_count."));
  PADDLE_ENFORCE_EQ(
      global_count.dtype(),
      DataType::INT64,
      common::errors::InvalidArgument("Please use int64 type in global_count."));

  const int64_t* cpu_local_count_data;
  const int64_t* cpu_global_count_data;
  DenseTensor cpu_local_count;
  if (local_count.place().GetType() == AllocationType::CPU) {
    cpu_local_count_data = local_count.data<int64_t>();
  } else {
    Copy(dev_ctx, local_count, CPUPlace(), true, &cpu_local_count);
    cpu_local_count_data = cpu_local_count.data<int64_t>();
  }

  int64_t global_count_len = 0;
  DenseTensor cpu_global_count;
  if (global_count.place().GetType() == AllocationType::CPU) {
    cpu_global_count_data = global_count.data<int64_t>();
    global_count_len = global_count.numel();
  } else {
    Copy(dev_ctx, global_count, CPUPlace(), true, &cpu_global_count);
    cpu_global_count_data = cpu_global_count.data<int64_t>();
    global_count_len = cpu_global_count.numel();
  }

  auto comm_ctx =
      static_cast<distributed::XCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("XCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));

  const int nranks = comm_ctx->GetSize();
  const auto in_feat = x.dims()[1];
  const auto n_expert = local_count.dims()[0] / nranks;
  int64_t fwd_count = 0;
  for (int64_t i = 0; i < global_count_len; ++i) {
    fwd_count += cpu_global_count_data[i];
  }

  out->Resize(make_ddim({fwd_count, in_feat}));
  dev_ctx.template Alloc<T>(out);

  std::vector<int64_t> expert_ptr(n_expert * nranks, 0);
  for (int64_t i = 1; i < n_expert * nranks; ++i) {
    expert_ptr[i] = expert_ptr[i - 1] + cpu_local_count_data[i - 1];
  }

  int64_t recv_ptr = 0;
  void* stream = reinterpret_cast<void*>(dev_ctx.stream());
  for (int64_t i = 0; i < n_expert; ++i) {
    comm_ctx->GroupStart();
    for (int j = 0; j < nranks; ++j) {
      int64_t idx = i + j * n_expert;
      if (cpu_local_count_data[idx]) {
        auto send_buf = distributed::GetPartialTensor(
            x,
            expert_ptr[idx] * in_feat,
            cpu_local_count_data[idx] * in_feat);
        comm_ctx->Send(send_buf, cpu_local_count_data[idx] * in_feat, j, stream);
      }
      if (cpu_global_count_data[idx]) {
        auto recv_buf = distributed::GetPartialTensor(
            *out, recv_ptr * in_feat, cpu_global_count_data[idx] * in_feat);
        comm_ctx->Recv(&recv_buf, cpu_global_count_data[idx] * in_feat, j, stream);
        recv_ptr += cpu_global_count_data[idx];
      }
    }
    comm_ctx->GroupEnd();
  }
}

template <typename T, typename Context>
void PartialAllGatherKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            int nranks,
                            int rank,
                            DenseTensor* out) {
  auto comm_ctx =
      static_cast<distributed::XCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("XCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  PADDLE_ENFORCE_EQ(nranks,
                    comm_ctx->GetSize(),
                    errors::InvalidArgument(
                        "nranks: %s should equal to %s", nranks, comm_ctx->GetSize()));
  PADDLE_ENFORCE_EQ(rank,
                    comm_ctx->GetRank(),
                    errors::InvalidArgument(
                        "rank: %s should equal to %s", rank, comm_ctx->GetRank()));
  PADDLE_ENFORCE_EQ(x.numel() % nranks,
                    0,
                    errors::InvalidArgument(
                        "The input numel (%d) must be divisible by nranks(%d)",
                        x.numel(),
                        nranks));

  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  int64_t send_numel = x.numel() / nranks;
  int64_t offset = send_numel * rank;
  auto send_buf = distributed::GetPartialTensor(x, offset, send_numel);
  void* stream = reinterpret_cast<void*>(dev_ctx.stream());
  comm_ctx->AllGather(out, send_buf, stream);
}

template <typename T, typename Context>
void MpAllReduceSumKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          DenseTensor* out) {
  out->Resize(x.dims());
  dev_ctx.template Alloc<T>(out);

  auto comm_ctx =
      static_cast<distributed::XCCLCommContext*>(dev_ctx.GetCommContext());
  PADDLE_ENFORCE_NE(
      comm_ctx,
      nullptr,
      errors::Unavailable("XCCLCommContext is nullptr, collective op should "
                          "has ring_id attr."));
  void* stream = reinterpret_cast<void*>(dev_ctx.stream());
  comm_ctx->AllReduce(out, x, phi::ccl::CCLReduceOp::SUM, stream);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(global_scatter,
                          musa,
                          ALL_LAYOUT,
                          phi::GlobalScatterKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::float16) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
  kernel->InputAt(2).SetDataType(phi::DataType::INT64);
}

PD_CUSTOM_KERNEL_REGISTER(partial_allgather,
                          musa,
                          ALL_LAYOUT,
                          phi::PartialAllGatherKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::bfloat16,
                          phi::float16) {}

PD_CUSTOM_KERNEL_REGISTER(mp_allreduce_sum,
                          musa,
                          ALL_LAYOUT,
                          phi::MpAllReduceSumKernel,
                          float,
                          double,
                          int,
                          int64_t,
                          phi::bfloat16,
                          phi::float16) {}

#undef STREAM_TYPE
#define STREAM_TYPE musaStream_t
