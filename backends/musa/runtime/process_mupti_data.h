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

#include <mupti.h>

#include <mutex>
#include <unordered_map>

#include "runtime/utils.h"

struct ActivityBuffer {
  ActivityBuffer(uint8_t* addr, size_t size) : addr(addr), valid_size(size) {}
  uint8_t* addr;
  size_t valid_size;
};

class Tracer {
 public:
  static Tracer& Instance() {
    static Tracer instance;
    return instance;
  }

  void AllocateBuffer(uint8_t** buffer, size_t* size);
  void ProduceBuffer(uint8_t* buffer, size_t valid_size);
  std::vector<ActivityBuffer> ConsumeBuffers();
  void ReleaseBuffer(uint8_t* buffer);

 private:
  Tracer() {}

  std::mutex activity_buffer_lock_;
  std::vector<ActivityBuffer> activity_buffers_;
};

class MuptiRuntimeCbidStr {
 public:
  static const MuptiRuntimeCbidStr& GetInstance() {
    static MuptiRuntimeCbidStr inst;
    return inst;
  }

  std::string RuntimeKind(MUpti_CallbackId cbid) const {
    auto iter = cbid_str_.find(cbid);
    if (iter == cbid_str_.end()) {
      return "musa Runtime API " + std::to_string(cbid);
    }
    return iter->second;
  }

 private:
  MuptiRuntimeCbidStr();

  std::unordered_map<MUpti_CallbackId, std::string> cbid_str_;
};
