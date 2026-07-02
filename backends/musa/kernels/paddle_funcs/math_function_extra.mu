/* Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/backends/custom/custom_context.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/math_function_impl.h"

namespace phi {
namespace funcs {

template struct SetConstant<phi::CustomContext, phi::float8_e4m3fn>;
template struct SetConstant<phi::CustomContext, phi::float8_e5m2>;
template struct SetConstant<phi::CustomContext, phi::float16>;
template struct SetConstant<phi::CustomContext, phi::bfloat16>;
template struct SetConstant<phi::CustomContext, float>;
template struct SetConstant<phi::CustomContext, double>;
template struct SetConstant<phi::CustomContext, uint8_t>;
template struct SetConstant<phi::CustomContext, uint16_t>;
template struct SetConstant<phi::CustomContext, uint32_t>;
template struct SetConstant<phi::CustomContext, uint64_t>;
template struct SetConstant<phi::CustomContext, int8_t>;
template struct SetConstant<phi::CustomContext, int>;
template struct SetConstant<phi::CustomContext, int16_t>;
template struct SetConstant<phi::CustomContext, int64_t>;
template struct SetConstant<phi::CustomContext, bool>;
template struct SetConstant<phi::CustomContext, phi::complex64>;
template struct SetConstant<phi::CustomContext, phi::complex128>;

#define DEFINE_CUSTOM_TRANS(RANK)                                      \
  template struct Transpose<phi::CustomContext, bool, RANK>;           \
  template struct Transpose<phi::CustomContext, uint8_t, RANK>;        \
  template struct Transpose<phi::CustomContext, uint16_t, RANK>;       \
  template struct Transpose<phi::CustomContext, uint32_t, RANK>;       \
  template struct Transpose<phi::CustomContext, uint64_t, RANK>;       \
  template struct Transpose<phi::CustomContext, float, RANK>;          \
  template struct Transpose<phi::CustomContext, double, RANK>;         \
  template struct Transpose<phi::CustomContext, phi::float8_e4m3fn, RANK>;  \
  template struct Transpose<phi::CustomContext, phi::float8_e5m2, RANK>;    \
  template struct Transpose<phi::CustomContext, phi::float16, RANK>;        \
  template struct Transpose<phi::CustomContext, phi::bfloat16, RANK>;       \
  template struct Transpose<phi::CustomContext, int8_t, RANK>;         \
  template struct Transpose<phi::CustomContext, int16_t, RANK>;        \
  template struct Transpose<phi::CustomContext, int32_t, RANK>;        \
  template struct Transpose<phi::CustomContext, int64_t, RANK>;        \
  template struct Transpose<phi::CustomContext, phi::complex64, RANK>; \
  template struct Transpose<phi::CustomContext, phi::complex128, RANK>;

DEFINE_CUSTOM_TRANS(1);
DEFINE_CUSTOM_TRANS(2);
DEFINE_CUSTOM_TRANS(3);
DEFINE_CUSTOM_TRANS(4);
DEFINE_CUSTOM_TRANS(5);
DEFINE_CUSTOM_TRANS(6);

#undef DEFINE_CUSTOM_TRANS

template struct ColwiseSum<phi::CustomContext, float>;
template struct ColwiseSum<phi::CustomContext, int>;
template struct ColwiseSum<phi::CustomContext, int64_t>;
template struct ColwiseSum<phi::CustomContext, double>;

template struct RowwiseSum<phi::CustomContext, float>;

template struct RowwiseMean<phi::CustomContext, float>;
template struct RowwiseMean<phi::CustomContext, double>;

}  // namespace funcs
}  // namespace phi
