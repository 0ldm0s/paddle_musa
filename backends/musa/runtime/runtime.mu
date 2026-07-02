// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

// - [add musa runtime api]
// - [add ccl and others api of Moorethread]

#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>

// #include "glog/logging.h"
#include "paddle/common/exception.h"
#include "paddle/phi/backends/device_ext.h"
#include "unsupported/Eigen/CXX11/Tensor"

#include "runtime/utils.h"
#include "runtime/musa_device_handler.h"  // NOLINT
#include "runtime/mccl_handler.h"
#include "runtime/mublas_handler.h"
#include "runtime/profiler_handler.h"

#define MEMORY_FRACTION 0.5f

static int global_current_device = 0;

C_Status Init() {
  std::cout << "custom_cpu plugin compiled with ";
#ifdef __clang__
  std::cout << "clang\n";
#else
  std::cout << "gcc\n";
#endif
  return C_SUCCESS;
}

C_Status InitDevice(const C_Device device) {
  global_current_device = device->id;
  return C_SUCCESS;
}

C_Status DestroyDevice(const C_Device device) { return C_SUCCESS; }

C_Status Finalize() { return C_SUCCESS; }

C_Status VisibleDevices(size_t *devices) { return C_SUCCESS; }

void InitPlugin(CustomRuntimeParams *params) {
  PADDLE_CUSTOM_RUNTIME_CHECK_VERSION(params);
  params->device_type = "musa";
  params->sub_device_type = "1.0.0";

  memset(reinterpret_cast<void *>(params->interface),
         0,
         sizeof(C_DeviceInterface));

  params->interface->initialize = Init;
  params->interface->finalize = Finalize;

  params->interface->init_device = musa::device::InitMusaDevice;
  params->interface->set_device = musa::device::SetMusaDevice;
  params->interface->get_device = musa::device::GetMusaDevice;
  params->interface->deinit_device = DestroyDevice;
  params->interface->get_device_count = musa::device::GetMusaDeviceCount;
  params->interface->get_device_list = musa::device::GetMusaDevicesList;

  params->interface->create_stream = musa::device::CreateMusaStream;
  params->interface->destroy_stream = musa::device::DestroyMusaStream;

  params->interface->create_event = musa::device::CreateMusaEvent;
  params->interface->destroy_event = musa::device::DestroyMusaEvent;
  params->interface->record_event = musa::device::RecordMusaEvent;

  params->interface->synchronize_device = musa::device::SyncMusaDevice;
  params->interface->synchronize_stream = musa::device::SyncMusaStream;
  params->interface->synchronize_event = musa::device::SyncMusaEvent;
  params->interface->stream_wait_event = musa::device::MusaStreamWaitEvent;

  params->interface->memory_copy_h2d = musa::device::MusaMemCpyH2D;
  params->interface->memory_copy_d2d = musa::device::MusaMemCpyD2D;
  params->interface->memory_copy_d2h = musa::device::MusaMemCpyD2H;
  params->interface->memory_copy_p2p = musa::device::MusaMemCpyP2P;
  params->interface->async_memory_copy_h2d = musa::device::MusaMemCpyH2DAsync;
  params->interface->async_memory_copy_d2d = musa::device::MusaMemCpyD2DAsync;
  params->interface->async_memory_copy_d2h = musa::device::MusaMemCpyD2HAsync;
  params->interface->async_memory_copy_p2p = musa::device::MusaMemCpyP2PAsync;
  params->interface->device_memory_allocate = musa::device::MusaMemAllocate;
  params->interface->host_memory_allocate = musa::device::MusaMemAllocateHost;
  params->interface->device_memory_deallocate = musa::device::MusaMemDeallocate;
  params->interface->host_memory_deallocate =
      musa::device::MusaMemDeallocateHost;

  params->interface->unified_memory_allocate =
      nullptr;  // TODO(jihong.zhong): fill it
  params->interface->unified_memory_deallocate =
      nullptr;  // TODO(jihong.zhong): fill it

  params->interface->device_memory_stats = musa::device::MusaDeviceMemStats;
  // params->interface->device_min_chunk_size = musa::device::MusaDeviceMinChunkSize;

  params->interface->get_compute_capability =
      musa::device::GetMusaComputeCapability;
  params->interface->get_device_properties =
      musa::device::GetMusaDeviceProperties;
  params->interface->get_runtime_version = musa::device::GetMusaRuntimeVersion;
  params->interface->get_driver_version = musa::device::GetMusaDriverVersion;
  params->interface->get_multi_process =
      musa::device::GetMusaMultiProcessorCount;
  params->interface->get_max_threads_per_mp =
      musa::device::GetMusaMaxThreadsPerMP;
  params->interface->get_max_threads_per_block =
      musa::device::GetMusaMaxThreadsPerBlock;
  params->interface->get_max_grid_dim_size =
      musa::device::GetMusaMaxGridDimSize;

  params->interface->init_eigen_device = musa::device::MusaInitEigenDevice;
  params->interface->destroy_eigen_device =
      musa::device::MusaDestroyEigenDevice;

  auto is_fp16_func = [](const C_Device device, bool *supported) -> C_Status {
    *supported = true;
    return C_SUCCESS;
  };
  auto is_bf16_func = is_fp16_func;
  // params->interface->is_float16_supported = is_fp16_func;
  // params->interface->is_bfloat16_supported = is_bf16_func;
  // //TODO(jihong.zhong): fix it

  params->interface->xccl_group_end = musa::mccl::McclGroupEnd;
  params->interface->xccl_group_start = musa::mccl::McclGroupStart;
  params->interface->xccl_get_unique_id_size = musa::mccl::McclGetUniqueIdSize;
  params->interface->xccl_get_unique_id = musa::mccl::McclGetUniqueId;
  params->interface->xccl_comm_init_rank = musa::mccl::McclCommInitRank;
  params->interface->xccl_destroy_comm = musa::mccl::McclDestroyComm;
  params->interface->xccl_all_reduce = musa::mccl::McclAllReduce;
  params->interface->xccl_all_gather = musa::mccl::McclAllGather;
  params->interface->xccl_broadcast = musa::mccl::McclBroadcast;
  params->interface->xccl_all_to_all = musa::mccl::McclAll2All;
  params->interface->xccl_reduce_scatter = musa::mccl::McclReduceScatter;
  params->interface->xccl_send = musa::mccl::McclSend;
  params->interface->xccl_recv = musa::mccl::McclRecv;
  params->interface->xccl_reduce = musa::mccl::McclReduce;
  params->interface->xccl_get_comm_name = [](C_CCLComm comm,
                                             char *comm_name) -> C_Status {
    static std::string name("PaddleWithMccl_" + musa::GetMcclVer());
    memcpy(comm_name, name.c_str(), name.size());
    return C_SUCCESS;
  };
  
  params->interface->init_blas_handle = musa::blas::InitBlasHandle;
  params->interface->init_blaslt_handle = musa::blas::InitBlasLtHandle;
  params->interface->destroy_blas_handle = musa::blas::DestroyBlasHandle;
  params->interface->destroy_blaslt_handle = musa::blas::DestroyBlasLtHandle;
  params->interface->blas_set_math_mode = musa::blas::BlasSetMathMode;

  params->interface->profiler_collect_trace_data = musa::profiler::ProfilerCollectData;
  params->interface->profiler_initialize = musa::profiler::ProfilerInitialize;
  params->interface->profiler_finalize = musa::profiler::ProfilerFinalize;
  params->interface->profiler_start_tracing = musa::profiler::ProfilerStart;
  params->interface->profiler_stop_tracing = musa::profiler::ProfilerStop;
  params->interface->profiler_prepare_tracing = musa::profiler::ProfilerPrepare;
}
