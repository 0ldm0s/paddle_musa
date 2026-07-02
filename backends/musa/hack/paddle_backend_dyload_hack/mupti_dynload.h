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
#pragma once

#include <musa.h>
#include <musa_occupancy.h>
#include <mupti.h>

#include <mutex>  // NOLINT

#include "musa_dynamic_loader.h"  // NOLINT
#include "paddle/phi/common/port.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace dynload {

extern std::once_flag mupti_dso_flag;
extern void *mupti_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cupti routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_MUPTI_WRAP(__name)                   \
  struct DynLoad__##__name {                                      \
    template <typename... Args>                                   \
    inline MUptiResult MUPTIAPI operator()(Args... args) {        \
      using muptiFunc = decltype(&::__name);                      \
      std::call_once(mupti_dso_flag, []() {                       \
        mupti_dso_handle = phi::dynload::GetMUPTIDsoHandle();     \
      });                                                         \
      static void *p_##__name = dlsym(mupti_dso_handle, #__name); \
      return reinterpret_cast<muptiFunc>(p_##__name)(args...);    \
    }                                                             \
  };                                                              \
  extern DynLoad__##__name __name

#define MUPTI_ROUTINE_EACH(__macro)           \
  __macro(muptiActivityEnable);               \
  __macro(muptiActivityDisable);              \
  __macro(muptiActivityRegisterCallbacks);    \
  __macro(muptiActivityGetAttribute);         \
  __macro(muptiActivitySetAttribute);         \
  __macro(muptiGetTimestamp);                 \
  __macro(muptiActivityGetNextRecord);        \
  __macro(muptiGetResultString);              \
  __macro(muptiActivityGetNumDroppedRecords); \
  __macro(muptiActivityFlushAll);             \
  __macro(muptiSubscribe);                    \
  __macro(muptiUnsubscribe);                  \
  __macro(muptiEnableCallback);               \
  __macro(muptiEnableDomain);                 \
  __macro(musaOccMaxActiveBlocksPerMultiprocessor);

MUPTI_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MUPTI_WRAP);

#undef DECLARE_DYNAMIC_LOAD_MUPTI_WRAP

}  // namespace dynload
}  // namespace phi
