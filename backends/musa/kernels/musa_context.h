// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// Modifications:
// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
// - [adapt to musa backend]

#pragma once

#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/backends/gpu/forwards.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"
#include "paddle/phi/backends/gpu/gpu_helper.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/attribute.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "mudnn_dynload.h"

namespace phi {
class muDnnWorkspaceHandle {
 public:
  inline muDnnWorkspaceHandle(Allocator* allocator, gpuStream_t stream)
      : allocator_(allocator), stream_(stream) {
    mtx_ = std::make_unique<std::mutex>();
  }

  inline void RunFunc(const std::function<void(void*)>& cudnn_func,
                      size_t required_workspace_bytes) {
    if (required_workspace_bytes > WorkspaceSize()) {
      ReallocWorkspace(required_workspace_bytes);
    }
    {
      std::lock_guard<std::mutex> guard(*mtx_);
      cudnn_func(allocation_ ? allocation_->ptr() : nullptr);
    }
  }
  inline size_t WorkspaceSize() {
    if (allocation_ == nullptr) {
      return 0;
    }
    return allocation_->size();
  }

  /*! \brief Thread which call RunFuncSync() would release gpu memory after
   *  running the function. Currently this function is only used when cudnn
   *  exhaustive searching and callers have to guarantee that the input function
   *  is host blocking */
  void RunFuncSync(const std::function<void(void*)>& cudnn_func,
                   size_t required_workspace_bytes,
                   bool use_cached_allocation = true);
  void ResetWorkspace();

  TEST_API void ReallocWorkspace(size_t required_workspace_bytes);

  muDnnWorkspaceHandle(muDnnWorkspaceHandle&&) = default;
  muDnnWorkspaceHandle& operator=(muDnnWorkspaceHandle&&) = delete;

 private:
  Allocator::AllocationPtr allocation_{nullptr};
  Allocator* allocator_{nullptr};  // Not owned
  gpuStream_t stream_{nullptr};    // Not owned
  std::unique_ptr<std::mutex> mtx_;
};

namespace {  // NOLINT
inline mudnnHandle_t dnn_handle_ = nullptr;
inline std::once_flag flag_dnn_;
inline void InitDnnHandle(mudnnHandle_t* handle,
                          gpuStream_t stream,
                          Place place) {
  if (phi::dynload::HasMUDNN()) {
    phi::dynload::mudnnCreate(handle, place.GetDeviceId());
    phi::dynload::mudnnSetStream(*handle, stream);
  } else {
    *handle = nullptr;
  }
}
}  // namespace

static mudnnHandle_t GetDnnHandle(gpuStream_t stream, GPUPlace place) {
  std::call_once(flag_dnn_, [&]() {
    if (!dnn_handle_) {
      InitDnnHandle(&dnn_handle_, stream, place);
    }
  });
  PADDLE_ENFORCE_NOT_NULL(
      dnn_handle_,
      common::errors::InvalidArgument(
          "The GPU dnn handle is nullptr. It must not be null."));
  return dnn_handle_;
}

inline muDnnWorkspaceHandle GetDnnWorkspace(Allocator* alloactor,
                                          const gpuStream_t& stream) {
  return muDnnWorkspaceHandle(alloactor, stream);
}

template <typename Context, typename T, size_t D>
static void RemovePaddingSlice(const Context& dev_ctx,
                               const DenseTensor* input,
                               DenseTensor* out,
                               const std::vector<int>& starts,
                               const std::vector<int>& axes) {
  auto& place = *dev_ctx.eigen_device();
  auto in_dims = input->dims();
  auto new_out_dims = out->dims();
  auto offsets = Eigen::DSizes<int64_t, D>();
  auto extents = Eigen::DSizes<int64_t, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = new_out_dims[i];
  }

  for (size_t i = 0; i < axes.size(); ++i) {
    int start = starts[i];
    if (start < 0) {
      start = start + in_dims[axes[i]];
    }
    start = std::max(start, 0);
    offsets[axes[i]] = start;
  }

  auto in_t = EigenTensor<T, D, Eigen::RowMajor>::From(*input);
  auto out_t = EigenTensor<T, D, Eigen::RowMajor>::From(*out, new_out_dims);
  funcs::EigenSlice<std::decay_t<decltype(place)>, T, D>::Eval(
      place, out_t, in_t, offsets, extents);
}
}  // namespace phi

// namespace musa {
// 
// class MusaContext {
//  public:
//   MusaContext() = default;
//   ~MusaContext();
// 
//   cuinferHandle_t getIxInferHandle();
// 
//  private:
//   cuinferHandle_t ixinfer_handle_{nullptr};
// };
// MusaContext* getContextInstance();
// 
// }  // namespace musa
// 
