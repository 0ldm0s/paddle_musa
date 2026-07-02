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
// Copyright (c) 2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
// - [register musa backend]

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/lu_unpack_grad_kernel_impl.h"
#include "paddle/phi/kernels/impl/lu_unpack_kernel_impl.h"
#include "paddle/phi/kernels/lu_unpack_grad_kernel.h"
#include "paddle/phi/kernels/lu_unpack_kernel.h"

namespace phi {

template <typename Context, typename T>
void MusaLUUnpack(const Context& dev_ctx,
                  const DenseTensor* lu,
                  DenseTensor* l,
                  DenseTensor* u) {
  const auto udims = lu->dims();
  l->Resize(udims);
  u->Resize(udims);
  const auto h = udims[udims.size() - 2];
  const auto w = udims[udims.size() - 1];

  dev_ctx.template Alloc<T>(l);
  auto* l_data = l->data<T>();
  funcs::ForRange<Context> lu_for_range(dev_ctx, lu->numel());
  funcs::TrilTriuCompute<T> tril_computer(
      lu->data<T>(), -1, true, h, w, l_data);
  lu_for_range(tril_computer);

  dev_ctx.template Alloc<T>(u);
  funcs::TrilTriuCompute<T> triu_computer(
      lu->data<T>(), 0, false, h, w, u->data<T>());
  lu_for_range(triu_computer);

  auto diag_size = std::min(h, w);
  DenseTensor rowtensor, rowtensor_dev;
  auto batchsize = product(slice_ddim(udims, 0, udims.size() - 2));
  if (udims.size() == 2) batchsize = std::max(static_cast<int>(batchsize), 1);

  arange<Context>(dev_ctx, &rowtensor, diag_size, batchsize, h);
  auto* index_data = rowtensor.data<int32_t>();
  if (dev_ctx.GetPlace().GetType() != AllocationType::CPU) {
    Copy(dev_ctx, rowtensor, dev_ctx.GetPlace(), false, &rowtensor_dev);
    index_data = rowtensor_dev.data<int32_t>();
  }

  funcs::ForRange<Context> diag_for_range(dev_ctx, rowtensor.numel());
  OneFunctor<T> functor(l_data, index_data, w, diag_size);
  diag_for_range(functor);
}

template <typename T, typename Context>
void MusaLUUnpackKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& pivots,
                        bool unpack_ludata,
                        bool unpack_pivots,
                        DenseTensor* pmat,
                        DenseTensor* l,
                        DenseTensor* u) {
  auto xdims = x.dims();
  int xrank = xdims.size();
  int64_t m = xdims[xrank - 2];
  int64_t n = xdims[xrank - 1];
  int64_t k = std::min(m, n);

  if (unpack_ludata) {
    dev_ctx.template Alloc<T>(l);
    dev_ctx.template Alloc<T>(u);

    if (x.numel() != 0) {
      DenseTensor tmp_l, tmp_u;
      MusaLUUnpack<Context, T>(dev_ctx, &x, &tmp_l, &tmp_u);

      if (m >= n) {
        Copy(dev_ctx, tmp_l, dev_ctx.GetPlace(), false, l);
        Tensor_narrow<Context, T>(dev_ctx, &tmp_u, u, 0, k, 0, k);
      } else {
        Copy(dev_ctx, tmp_u, dev_ctx.GetPlace(), false, u);
        Tensor_narrow<Context, T>(dev_ctx, &tmp_l, l, 0, k, 0, k);
      }
    }
  }

  if (unpack_pivots) {
    dev_ctx.template Alloc<T>(pmat);
    if (x.numel() == 0 || pivots.numel() == 0) {
      auto pmat_dims = pmat->dims();
      int64_t columns = pmat_dims[pmat_dims.size() - 1];
      if (columns == 0) return;

      T* pmat_data = pmat->data<T>();
      funcs::SetConstant<Context, T> set_zero;
      set_zero(dev_ctx, pmat, static_cast<T>(0));
      int64_t rows = pmat->numel() / columns;
      funcs::ForRange<Context> for_range(dev_ctx, rows);
      LuUnpackEyeFunctor<T> functor(columns, pmat_data);
      for_range(functor);
      return;
    }

    PADDLE_ENFORCE_EQ(
        pivots.dtype(),
        DataType::INT32,
        common::errors::InvalidArgument(
            "The pivots of lu_unpack must be of type int32, but received [%s].",
            pivots.dtype()));

    Unpack_Pivot<Context, T>(dev_ctx, pivots, pmat, m, k);
  }
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(lu_unpack,
                          musa,
                          ALL_LAYOUT,
                          phi::MusaLUUnpackKernel,
                          float,
                          double,
                          phi::complex64,
                          phi::complex128) {}

PD_CUSTOM_KERNEL_REGISTER(lu_unpack_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::LUUnpackGradKernel,
                          float,
                          double,
                          phi::complex64,
                          phi::complex128) {}
