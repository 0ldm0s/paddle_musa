// Copyright (c) 2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
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


#include "runtime/profiler_handler.h"
#include "paddle/phi/core/os_info.h"
#include "mupti_dynload.h" // NOLINT

namespace musa::profiler {

struct ProfilerContext {
  uint64_t mupti_start_ns = 0;
  uint64_t host_start_ns = 0;
};

inline static uint64_t ToHostTime(uint64_t timestamp, uint64_t mupti_start_ns, uint64_t host_start_ns) {
  if (timestamp < mupti_start_ns || mupti_start_ns == 0 || host_start_ns == 0) {
    return timestamp;
  }
  return host_start_ns + (timestamp - mupti_start_ns);
}

inline static const char* MemcpyKind(uint8_t kind) {
    static const std::unordered_map<uint8_t, const char *> MEM_CPY_KIND = {
        {MUPTI_ACTIVITY_MEMCPY_KIND_HTOD, "MEMCPY_HtoD"},
        {MUPTI_ACTIVITY_MEMCPY_KIND_DTOH, "MEMCPY_DtoH"},
        {MUPTI_ACTIVITY_MEMCPY_KIND_HTOA, "MEMCPY_HtoA"},
        {MUPTI_ACTIVITY_MEMCPY_KIND_ATOH, "MEMCPY_AtoH"},
        {MUPTI_ACTIVITY_MEMCPY_KIND_ATOA, "MEMCPY_AtoA"},
        {MUPTI_ACTIVITY_MEMCPY_KIND_ATOD, "MEMCPY_AtoD"},
        {MUPTI_ACTIVITY_MEMCPY_KIND_DTOA, "MEMCPY_DtoA"},
        {MUPTI_ACTIVITY_MEMCPY_KIND_DTOD, "MEMCPY_DtoD"},
        {MUPTI_ACTIVITY_MEMCPY_KIND_HTOH, "MEMCPY_HtoH"},
        {MUPTI_ACTIVITY_MEMCPY_KIND_PTOP, "MEMCPY_PtoP"}
      };
    
    auto iter = MEM_CPY_KIND.find(kind);
    return iter != MEM_CPY_KIND.end()? iter->second: "MEMCPY"; 
}

static std::unordered_map<uint32_t, uint64_t> CreateThreadIdMapping() {
  std::unordered_map<uint32_t, uint64_t> mapping;
  std::unordered_map<uint64_t, phi::ThreadId> ids = phi::GetAllThreadIds();
  for (const auto& id : ids) {
    mapping[id.second.cupti_tid] = id.second.sys_tid;
  }
  return mapping;
}

void BufferRequestedCallback(uint8_t **buffer,
                             size_t *size,
                             size_t *max_num_records) {
  Tracer::Instance().AllocateBuffer(buffer, size);
  *max_num_records = 0;
}

void BufferCompletedCallback(MUcontext ctx,
                             uint32_t stream_id,
                             uint8_t *buffer,
                             size_t size,
                             size_t valid_size) {
  Tracer::Instance().ProduceBuffer(buffer, valid_size);
  size_t dropped = 0;
  MUPTI_CHECK(phi::dynload::muptiActivityGetNumDroppedRecords(ctx, stream_id, &dropped));
  if (dropped != 0) {
    LOG(WARNING) << "stream " << stream_id << " dropped " << dropped
                 << " activity records in paddle_musa";
  }
}

inline static void AddDeviceTraceEvent(C_Profiler prof, void* event) {
  phi::DeviceTraceEvent de =
      *reinterpret_cast<phi::DeviceTraceEvent*>(event);
  reinterpret_cast<phi::TraceEventCollector*>(prof)
      ->AddDeviceEvent(std::move(de));
}

inline static void AddRuntimeTraceEvent(C_Profiler prof, void* event) {
  phi::RuntimeTraceEvent re =
      *reinterpret_cast<phi::RuntimeTraceEvent*>(event);
  reinterpret_cast<phi::TraceEventCollector*>(prof)
      ->AddRuntimeEvent(std::move(re));
}

static void AddKernelRecord(const MUpti_ActivityKernel4* kernel,
                            uint64_t mupti_start_ns,
                            uint64_t host_start_ns,
                            C_Profiler collector) {
  if (kernel->start < mupti_start_ns) {
    return;
  }

  auto demangle = [](std::string name) -> std::string {
      int status = -4;
      
      std::unique_ptr<char, void (*)(void*)> res {
          abi::__cxa_demangle(name.c_str(), NULL, NULL, &status), std::free};
      return (status == 0) ? res.get() : name;
  };

  phi::DeviceTraceEvent event;
  event.name = demangle(kernel->name);
  event.type = phi::TracerEventType::Kernel;
  event.start_ns = ToHostTime(kernel->start, mupti_start_ns, host_start_ns);
  event.end_ns = ToHostTime(kernel->end, mupti_start_ns, host_start_ns);
  event.device_id = kernel->deviceId;
  event.context_id = kernel->contextId;
  event.stream_id = kernel->streamId;
  event.correlation_id = kernel->correlationId;
  event.kernel_info.block_x = kernel->blockX;
  event.kernel_info.block_y = kernel->blockY;
  event.kernel_info.block_z = kernel->blockZ;
  event.kernel_info.grid_x = kernel->gridX;
  event.kernel_info.grid_y = kernel->gridY;
  event.kernel_info.grid_z = kernel->gridZ;
  event.kernel_info.dynamic_shared_memory = kernel->dynamicSharedMemory;
  event.kernel_info.static_shared_memory = kernel->staticSharedMemory;
  event.kernel_info.registers_per_thread = kernel->registersPerThread;
  event.kernel_info.local_memory_per_thread = kernel->localMemoryPerThread;
  event.kernel_info.local_memory_total = kernel->localMemoryTotal;
  event.kernel_info.queued = kernel->queued;
  event.kernel_info.submitted = kernel->submitted;
  event.kernel_info.completed = kernel->completed;

  musa::profiler::AddDeviceTraceEvent(collector, &event);
}

static void AddMemcpyRecord(const MUpti_ActivityMemcpy* memcpy,
                            uint64_t mupti_start_ns,
                            uint64_t host_start_ns,
                            C_Profiler collector) {
  if (memcpy->start < mupti_start_ns) {
    return;
  }
  phi::DeviceTraceEvent event;
  event.name = MemcpyKind(memcpy->copyKind);
  event.type = phi::TracerEventType::Memcpy;
  event.start_ns = ToHostTime(memcpy->start, mupti_start_ns, host_start_ns);
  event.end_ns = ToHostTime(memcpy->end, mupti_start_ns, host_start_ns);
  event.device_id = memcpy->deviceId;
  event.context_id = memcpy->contextId;
  event.stream_id = memcpy->streamId;
  event.correlation_id = memcpy->correlationId;
  event.memcpy_info.num_bytes = memcpy->bytes;

  snprintf(event.memcpy_info.src_kind,
           phi::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy->srcKind));
  snprintf(event.memcpy_info.dst_kind,
           phi::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy->dstKind));

  musa::profiler::AddDeviceTraceEvent(collector, &event);

}

static void AddMemcpy2Record(const MUpti_ActivityMemcpy2* memcpy2,
                      uint64_t mupti_start_ns,
                      uint64_t host_start_ns,
                      C_Profiler collector) {
  if (memcpy2->start < mupti_start_ns) {
    return;
  }
  phi::DeviceTraceEvent event;
  event.name = MemcpyKind(memcpy2->copyKind);
  event.type = phi::TracerEventType::Memcpy;
  event.start_ns = ToHostTime(memcpy2->start, mupti_start_ns, host_start_ns);
  event.end_ns = ToHostTime(memcpy2->end, mupti_start_ns, host_start_ns);
  event.device_id = memcpy2->deviceId;
  event.context_id = memcpy2->contextId;
  event.stream_id = memcpy2->streamId;
  event.correlation_id = memcpy2->correlationId;
  event.memcpy_info.num_bytes = memcpy2->bytes;

  snprintf(event.memcpy_info.src_kind,
           phi::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy2->srcKind));
  snprintf(event.memcpy_info.dst_kind,
           phi::kMemKindMaxLen,
           "%s",
           MemcpyKind(memcpy2->dstKind));

  musa::profiler::AddDeviceTraceEvent(collector, &event);
}

static void AddMemsetRecord(const MUpti_ActivityMemset* memset,
                            uint64_t mupti_start_ns,
                            uint64_t host_start_ns,
                            C_Profiler collector) {
  if (memset->start < mupti_start_ns) {
    return;
  }
  phi::DeviceTraceEvent event;
  event.name = "MEMSET";
  event.type = phi::TracerEventType::Memset;
  event.start_ns = ToHostTime(memset->start, mupti_start_ns, host_start_ns);
  event.end_ns = ToHostTime(memset->end, mupti_start_ns, host_start_ns);
  event.device_id = memset->deviceId;
  event.context_id = memset->contextId;
  event.stream_id = memset->streamId;
  event.correlation_id = memset->correlationId;
  event.memset_info.num_bytes = memset->bytes;

  event.memset_info.value = memset->value;

  musa::profiler::AddDeviceTraceEvent(collector, &event);
}

static void AddApiRecord(const MUpti_ActivityAPI* api,
                         uint64_t mupti_start_ns,
                         uint64_t host_start_ns,
                         const std::unordered_map<uint32_t, uint64_t> tid_mapping,
                         C_Profiler collector) {
  if (api->start < mupti_start_ns) {
    return;
  }
  phi::RuntimeTraceEvent event;
  event.name = MuptiRuntimeCbidStr::GetInstance().RuntimeKind(api->cbid);
  event.start_ns = ToHostTime(api->start, mupti_start_ns, host_start_ns);
  event.end_ns = ToHostTime(api->end, mupti_start_ns, host_start_ns);
  event.process_id = phi::GetProcessId();
  uint64_t tid = gettid();
  auto iter = tid_mapping.find(api->threadId);
  if (iter == tid_mapping.end()) {
  } else {
    tid = iter->second;
  }

  event.thread_id = tid;

  event.correlation_id = api->correlationId;
  event.callback_id = api->cbid;

  event.type = phi::TracerEventType::CudaRuntime;
  musa::profiler::AddRuntimeTraceEvent(collector, &event);
}

static void ProcessMuptiActivityRecord(
    const MUpti_Activity* record,
    uint64_t mupti_start_ns,
    uint64_t host_start_ns,
    const std::unordered_map<uint32_t, uint64_t> tid_mapping,
    C_Profiler collector) {
   switch (record->kind) {
     case MUPTI_ACTIVITY_KIND_KERNEL:
     case MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
       AddKernelRecord(reinterpret_cast<const MUpti_ActivityKernel4*>(record),
                       mupti_start_ns,
                       host_start_ns,
                       collector);
       break;
     case MUPTI_ACTIVITY_KIND_MEMCPY:
       AddMemcpyRecord(reinterpret_cast<const MUpti_ActivityMemcpy*>(record),
                       mupti_start_ns,
                       host_start_ns,
                       collector);
       break;
     case MUPTI_ACTIVITY_KIND_MEMCPY2:
       AddMemcpy2Record(reinterpret_cast<const MUpti_ActivityMemcpy2*>(record),
                        mupti_start_ns,
                        host_start_ns,
                        collector);
       break;
     case MUPTI_ACTIVITY_KIND_MEMSET:
       AddMemsetRecord(reinterpret_cast<const MUpti_ActivityMemset*>(record),
                       mupti_start_ns,
                       host_start_ns,
                       collector);
       break;
     case MUPTI_ACTIVITY_KIND_DRIVER:
     case MUPTI_ACTIVITY_KIND_RUNTIME:
       AddApiRecord(reinterpret_cast<const MUpti_ActivityAPI*>(record),
                    mupti_start_ns,
                    host_start_ns,
                    tid_mapping,
                    collector);
       break;
     default:
       break;
   }
}


static int ProcessMuptiActivity(C_Profiler prof, uint64_t mupti_start_ns, uint64_t host_start_ns) {
  int record_cnt = 0;
  MUPTI_CHECK(phi::dynload::muptiActivityFlushAll(MUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  auto mapping = musa::profiler::CreateThreadIdMapping();
  std::vector<ActivityBuffer> buffers = Tracer::Instance().ConsumeBuffers();
  for (auto &buffer : buffers) {
    if (buffer.addr == nullptr || buffer.valid_size == 0) {
      continue;
    }
    MUpti_Activity *record = nullptr;
    do {
      MUptiResult status =
          phi::dynload::muptiActivityGetNextRecord(buffer.addr, buffer.valid_size, &record);
      if (status == MUPTI_SUCCESS) {
        ProcessMuptiActivityRecord(record, mupti_start_ns, host_start_ns, mapping, prof);
        ++record_cnt;
      } else if (status == MUPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        MUPTI_CHECK(status);
      }
    } while (true);

    Tracer::Instance().ReleaseBuffer(buffer.addr);
  }
  return record_cnt;
}

C_Status ProfilerInitialize(C_Profiler prof, void **user_data) {
  if (user_data != nullptr) {
    *user_data = new ProfilerContext();
  }
  return C_SUCCESS;
}

C_Status ProfilerFinalize(C_Profiler prof, void *user_data) {
  auto buffers = Tracer::Instance().ConsumeBuffers();
  for (auto& buffer : buffers) {
    if (buffer.addr != nullptr) {
      Tracer::Instance().ReleaseBuffer(buffer.addr);
    }
  }
  delete reinterpret_cast<ProfilerContext*>(user_data);
  return C_SUCCESS;
}

C_Status ProfilerPrepare(C_Profiler prof, void *user_data) {
  MUPTI_CHECK(phi::dynload::muptiActivityRegisterCallbacks(BufferRequestedCallback,
                                             BufferCompletedCallback));
  MUPTI_CHECK(phi::dynload::muptiActivityEnable(MUPTI_ACTIVITY_KIND_KERNEL));
  MUPTI_CHECK(phi::dynload::muptiActivityEnable(MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  MUPTI_CHECK(phi::dynload::muptiActivityEnable(MUPTI_ACTIVITY_KIND_MEMCPY));
  MUPTI_CHECK(phi::dynload::muptiActivityEnable(MUPTI_ACTIVITY_KIND_DRIVER));
  MUPTI_CHECK(phi::dynload::muptiActivityEnable(MUPTI_ACTIVITY_KIND_RUNTIME));
  MUPTI_CHECK(phi::dynload::muptiActivityEnable(MUPTI_ACTIVITY_KIND_MEMSET));
  VLOG(3) << "enable mupti activity in paddle_musa";
  return C_SUCCESS;
}

C_Status ProfilerStart(C_Profiler prof, void *user_data) {
  auto buffers = Tracer::Instance().ConsumeBuffers();
  for (auto& buffer : buffers) {
    if (buffer.addr != nullptr) {
      Tracer::Instance().ReleaseBuffer(buffer.addr);
    }
  }
  if (user_data != nullptr) {
    auto* ctx = reinterpret_cast<ProfilerContext*>(user_data);
    MUPTI_CHECK(phi::dynload::muptiGetTimestamp(&ctx->mupti_start_ns));
    ctx->host_start_ns = phi::PosixInNsec();
  }
  return C_SUCCESS;
}

C_Status ProfilerStop(C_Profiler prof, void *user_data) {
  MUPTI_CHECK(phi::dynload::muptiActivityDisable(MUPTI_ACTIVITY_KIND_KERNEL));
  MUPTI_CHECK(phi::dynload::muptiActivityDisable(MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  MUPTI_CHECK(phi::dynload::muptiActivityDisable(MUPTI_ACTIVITY_KIND_MEMCPY));
  MUPTI_CHECK(phi::dynload::muptiActivityDisable(MUPTI_ACTIVITY_KIND_DRIVER));
  MUPTI_CHECK(phi::dynload::muptiActivityDisable(MUPTI_ACTIVITY_KIND_MEMSET));
  MUPTI_CHECK(phi::dynload::muptiActivityDisable(MUPTI_ACTIVITY_KIND_RUNTIME));
  VLOG(3) << "disable mupti activity in paddle_musa";
  return C_SUCCESS;
}

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t tracing_start_ns,
                             void *user_data) {
  uint64_t mupti_start_ns = tracing_start_ns;
  uint64_t host_start_ns = tracing_start_ns;
  if (user_data != nullptr) {
    auto* ctx = reinterpret_cast<ProfilerContext*>(user_data);
    if (ctx->mupti_start_ns != 0) {
      mupti_start_ns = ctx->mupti_start_ns;
      host_start_ns = ctx->host_start_ns;
    }
  }
  ProcessMuptiActivity(prof, mupti_start_ns, host_start_ns);
  return C_SUCCESS;
}

}
