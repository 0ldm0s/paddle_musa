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

#include <musa_runtime.h>
#include <mupti.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "glog/logging.h"
#include "paddle/fluid/platform/profiler/trace_event_collector.h"
#include "paddle/phi/api/profiler/trace_event_collector.h"
#include "paddle/phi/backends/custom/cuda_graph.h"
#include "paddle/phi/core/platform/profiler/utils.h"
// #include "passes/pattern_passes.h"
#include "runtime/process_mupti_data.h"  //NOLINT
#include "unsupported/Eigen/CXX11/Tensor"
#include "runtime/utils.h"


// namespace paddle {
// namespace platform {
// namespace details {
// 
// float CalculateEstOccupancy(uint32_t DeviceId,
//                             uint16_t RegistersPerThread,
//                             int32_t StaticSharedMemory,
//                             int32_t DynamicSharedMemory,
//                             int32_t BlockX,
//                             int32_t BlockY,
//                             int32_t BlockZ,
//                             float BlocksPerSm);

// void ProcessMuptiActivityRecord(
//     const MUpti_Activity* record,
//     uint64_t start_ns,
//     const std::unordered_map<uint32_t, uint64_t> tid_mapping,
//     TraceEventCollector* collector);
// 
// }  // namespace details
// }  // namespace platform
// }  // namespace paddle
// 



namespace musa::profiler {

C_Status ProfilerInitialize(C_Profiler prof, void **user_data);

C_Status ProfilerFinalize(C_Profiler prof, void *user_data);

C_Status ProfilerPrepare(C_Profiler prof, void *user_data);

C_Status ProfilerStart(C_Profiler prof, void *user_data);

C_Status ProfilerStop(C_Profiler prof, void *user_data);

C_Status ProfilerCollectData(C_Profiler prof,
                             uint64_t start_ns,
                             void *user_data);

}
