// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
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
#include "runtime/process_mupti_data.h"

void* AlignedMalloc(size_t size, size_t alignment) {
  assert(alignment >= sizeof(void*) && (alignment & (alignment - 1)) == 0);
  size = (size + alignment - 1) / alignment * alignment;
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
  void* aligned_mem = nullptr;
  if (posix_memalign(&aligned_mem, alignment, size) != 0) {
    aligned_mem = nullptr;
  }
  return aligned_mem;
#else
  void* mem = malloc(size + alignment);  // NOLINT
  if (mem == nullptr) {
    return nullptr;
  }
  size_t adjust = alignment - reinterpret_cast<uint64_t>(mem) % alignment;
  void* aligned_mem = reinterpret_cast<char*>(mem) + adjust;
  *(reinterpret_cast<void**>(aligned_mem) - 1) = mem;
  assert(reinterpret_cast<uint64_t>(aligned_mem) % alignment == 0);
  return aligned_mem;
#endif
}

void AlignedFree(void* mem_ptr) {
#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
  free(mem_ptr);
#else
  if (mem_ptr) {
    free(*(reinterpret_cast<void**>(mem_ptr) - 1));
  }
#endif
}

void Tracer::AllocateBuffer(uint8_t** buffer, size_t* size) {
  constexpr size_t kBufferSize = 1 << 23;  // 8 MB
  constexpr size_t kBufferAlignSize = 8;
  *buffer =
      reinterpret_cast<uint8_t*>(AlignedMalloc(kBufferSize, kBufferAlignSize));
  *size = kBufferSize;
}

void Tracer::ProduceBuffer(uint8_t* buffer, size_t valid_size) {
  std::lock_guard<std::mutex> guard(activity_buffer_lock_);
  activity_buffers_.emplace_back(buffer, valid_size);
}

std::vector<ActivityBuffer> Tracer::ConsumeBuffers() {
  std::vector<ActivityBuffer> buffers;
  {
    std::lock_guard<std::mutex> guard(activity_buffer_lock_);
    buffers.swap(activity_buffers_);
  }
  return buffers;
}

void Tracer::ReleaseBuffer(uint8_t* buffer) { AlignedFree(buffer); }

MuptiRuntimeCbidStr::MuptiRuntimeCbidStr() {
#define REGISTER_RUNTIME_CBID_STR(cbid) \
  cbid_str_[MUPTI_RUNTIME_TRACE_CBID_##cbid] = #cbid
  REGISTER_RUNTIME_CBID_STR(musaDriverGetVersion_v3020);
  REGISTER_RUNTIME_CBID_STR(musaRuntimeGetVersion_v3020);
  REGISTER_RUNTIME_CBID_STR(musaGetDeviceCount_v3020);
  REGISTER_RUNTIME_CBID_STR(musaGetDeviceProperties_v3020);
  REGISTER_RUNTIME_CBID_STR(musaChooseDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(musaGetLastError_v3020);
  REGISTER_RUNTIME_CBID_STR(musaPeekAtLastError_v3020);
  REGISTER_RUNTIME_CBID_STR(musaLaunch_v3020);
  REGISTER_RUNTIME_CBID_STR(musaFuncSetCacheConfig_v3020);
  REGISTER_RUNTIME_CBID_STR(musaFuncGetAttributes_v3020);
  REGISTER_RUNTIME_CBID_STR(musaSetDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(musaGetDevice_v3020);
  REGISTER_RUNTIME_CBID_STR(musaSetValidDevices_v3020);
  REGISTER_RUNTIME_CBID_STR(musaSetDeviceFlags_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMalloc_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMallocPitch_v3020);
  REGISTER_RUNTIME_CBID_STR(musaFree_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMallocArray_v3020);
  REGISTER_RUNTIME_CBID_STR(musaFreeArray_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMallocHost_v3020);
  REGISTER_RUNTIME_CBID_STR(musaFreeHost_v3020);
  REGISTER_RUNTIME_CBID_STR(musaHostAlloc_v3020);
  REGISTER_RUNTIME_CBID_STR(musaHostGetDevicePointer_v3020);
  REGISTER_RUNTIME_CBID_STR(musaHostGetFlags_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemGetInfo_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpy_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpy2D_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpyToArray_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpy2DToArray_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpyToSymbol_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpyFromSymbol_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpyAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpy2DAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpyToSymbolAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpyFromSymbolAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemset_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemset2D_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemsetAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemset2DAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(musaGetSymbolAddress_v3020);
  REGISTER_RUNTIME_CBID_STR(musaGetSymbolSize_v3020);
  REGISTER_RUNTIME_CBID_STR(musaBindTexture_v3020);
  REGISTER_RUNTIME_CBID_STR(musaBindTexture2D_v3020);
  REGISTER_RUNTIME_CBID_STR(musaBindTextureToArray_v3020);
  REGISTER_RUNTIME_CBID_STR(musaUnbindTexture_v3020);
  REGISTER_RUNTIME_CBID_STR(musaStreamCreate_v3020);
  REGISTER_RUNTIME_CBID_STR(musaStreamDestroy_v3020);
  REGISTER_RUNTIME_CBID_STR(musaStreamSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(musaStreamQuery_v3020);
  REGISTER_RUNTIME_CBID_STR(musaEventCreate_v3020);
  REGISTER_RUNTIME_CBID_STR(musaEventCreateWithFlags_v3020);
  REGISTER_RUNTIME_CBID_STR(musaEventRecord_v3020);
  REGISTER_RUNTIME_CBID_STR(musaEventDestroy_v3020);
  REGISTER_RUNTIME_CBID_STR(musaEventSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(musaEventQuery_v3020);
  REGISTER_RUNTIME_CBID_STR(musaEventElapsedTime_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMalloc3D_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMalloc3DArray_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemset3D_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemset3DAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpy3D_v3020);
  REGISTER_RUNTIME_CBID_STR(musaMemcpy3DAsync_v3020);
  REGISTER_RUNTIME_CBID_STR(musaStreamWaitEvent_v3020);
  REGISTER_RUNTIME_CBID_STR(musaPointerGetAttributes_v4000);
  REGISTER_RUNTIME_CBID_STR(musaHostRegister_v4000);
  REGISTER_RUNTIME_CBID_STR(musaHostUnregister_v4000);
  REGISTER_RUNTIME_CBID_STR(musaDeviceCanAccessPeer_v4000);
  REGISTER_RUNTIME_CBID_STR(musaDeviceEnablePeerAccess_v4000);
  REGISTER_RUNTIME_CBID_STR(musaDeviceDisablePeerAccess_v4000);
  REGISTER_RUNTIME_CBID_STR(musaMemcpyPeer_v4000);
  REGISTER_RUNTIME_CBID_STR(musaMemcpyPeerAsync_v4000);
  REGISTER_RUNTIME_CBID_STR(musaMemcpy3DPeer_v4000);
  REGISTER_RUNTIME_CBID_STR(musaMemcpy3DPeerAsync_v4000);
  REGISTER_RUNTIME_CBID_STR(musaDeviceReset_v3020);
  REGISTER_RUNTIME_CBID_STR(musaDeviceSynchronize_v3020);
  REGISTER_RUNTIME_CBID_STR(musaDeviceGetLimit_v3020);
  REGISTER_RUNTIME_CBID_STR(musaDeviceSetLimit_v3020);
  REGISTER_RUNTIME_CBID_STR(musaDeviceGetCacheConfig_v3020);
  REGISTER_RUNTIME_CBID_STR(musaDeviceSetCacheConfig_v3020);
  REGISTER_RUNTIME_CBID_STR(musaProfilerInitialize_v4000);
  REGISTER_RUNTIME_CBID_STR(musaProfilerStart_v4000);
  REGISTER_RUNTIME_CBID_STR(musaProfilerStop_v4000);
  REGISTER_RUNTIME_CBID_STR(musaDeviceGetByPCIBusId_v4010);
  REGISTER_RUNTIME_CBID_STR(musaDeviceGetPCIBusId_v4010);
  REGISTER_RUNTIME_CBID_STR(musaIpcGetEventHandle_v4010);
  REGISTER_RUNTIME_CBID_STR(musaIpcOpenEventHandle_v4010);
  REGISTER_RUNTIME_CBID_STR(musaIpcGetMemHandle_v4010);
  REGISTER_RUNTIME_CBID_STR(musaIpcOpenMemHandle_v4010);
  REGISTER_RUNTIME_CBID_STR(musaIpcCloseMemHandle_v4010);
  REGISTER_RUNTIME_CBID_STR(musaFuncSetSharedMemConfig_v4020);
  REGISTER_RUNTIME_CBID_STR(musaDeviceGetSharedMemConfig_v4020);
  REGISTER_RUNTIME_CBID_STR(musaDeviceSetSharedMemConfig_v4020);
  REGISTER_RUNTIME_CBID_STR(musaStreamAddCallback_v5000);
  REGISTER_RUNTIME_CBID_STR(musaStreamCreateWithFlags_v5000);
  REGISTER_RUNTIME_CBID_STR(musaDeviceGetAttribute_v5000);
  REGISTER_RUNTIME_CBID_STR(musaStreamDestroy_v5050);
  REGISTER_RUNTIME_CBID_STR(musaStreamCreateWithPriority_v5050);
  REGISTER_RUNTIME_CBID_STR(musaStreamGetPriority_v5050);
  REGISTER_RUNTIME_CBID_STR(musaStreamGetFlags_v5050);
  REGISTER_RUNTIME_CBID_STR(musaDeviceGetStreamPriorityRange_v5050);
  REGISTER_RUNTIME_CBID_STR(musaMallocManaged_v6000);
  REGISTER_RUNTIME_CBID_STR(
      musaOccupancyMaxActiveBlocksPerMultiprocessor_v6000);
  REGISTER_RUNTIME_CBID_STR(musaStreamAttachMemAsync_v6000);
  REGISTER_RUNTIME_CBID_STR(
      musaOccupancyMaxActiveBlocksPerMultiprocessor_v6050);
  REGISTER_RUNTIME_CBID_STR(musaLaunchKernel_v7000);
  REGISTER_RUNTIME_CBID_STR(musaGetDeviceFlags_v7000);
  REGISTER_RUNTIME_CBID_STR(
      musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_v7000);
  REGISTER_RUNTIME_CBID_STR(musaMemRangeGetAttribute_v8000);
  REGISTER_RUNTIME_CBID_STR(musaMemRangeGetAttributes_v8000);
  REGISTER_RUNTIME_CBID_STR(musaLaunchCooperativeKernel_v9000);
  REGISTER_RUNTIME_CBID_STR(musaLaunchCooperativeKernelMultiDevice_v9000);
  REGISTER_RUNTIME_CBID_STR(musaFuncSetAttribute_v9000);
  REGISTER_RUNTIME_CBID_STR(musaGraphLaunch_v10000);
  REGISTER_RUNTIME_CBID_STR(musaStreamSetAttribute_v11000);
  REGISTER_RUNTIME_CBID_STR(musaMallocAsync_v11020);
  REGISTER_RUNTIME_CBID_STR(musaFreeAsync_v11020);
#undef REGISTER_RUNTIME_CBID_STR
}
