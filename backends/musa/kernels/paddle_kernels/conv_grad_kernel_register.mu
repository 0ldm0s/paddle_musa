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

// Modifications:
// Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
// rights reserved.
// - [register musa backend]

#include <type_traits>

#include "paddle/phi/kernels/conv_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/padding.h"
#include "paddle/phi/kernels/impl/conv_grad_kernel_impl.h"
#include "paddle/phi/kernels/impl/slow_conv_kernel_impl.cuh"

#include "kernels/kernels_utils.h"
#include "kernels/musa_context.h"

namespace phi {

template <typename T, typename Context>
bool PrepareMudnnConvGradInput(const Context& dev_ctx,
                               const DenseTensor& input,
                               const DenseTensor& filter,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings_t,
                               const std::string& padding_algorithm,
                               const std::vector<int>& dilations_t,
                               const std::string& data_format,
                               DenseTensor* transformed_input,
                               DenseTensor* transformed_input_grad,
                               std::vector<int>* padding_common,
                               std::vector<int>* dilations,
                               std::vector<int>* input_pad,
                               bool* is_asymmetric_pad) {
  *dilations = dilations_t;
  std::vector<int> paddings = paddings_t;
  *is_asymmetric_pad = false;

  auto in_dims = input.dims();
  auto filter_dims = filter.dims();
  DDim in_data_dims;
  const DataLayout data_layout = StringToDataLayout(data_format);
  if (data_layout != DataLayout::NHWC) {
    in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
  } else {
    in_data_dims = slice_ddim(in_dims, 1, in_dims.size() - 1);
  }

  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, dilations, padding_algorithm, in_data_dims, strides, ksize);

  const int data_dim = strides.size();
  padding_common->assign(data_dim, 0);
  input_pad->assign(input.dims().size() * 2, 0);
  if (paddings.size() == strides.size()) {
    *padding_common = paddings;
    transformed_input->ShareDataWith(input);
    transformed_input_grad->Resize(input.dims());
    return true;
  }
  if (paddings.size() != strides.size() * 2) {
    return false;
  }

  const bool is_symmetric = funcs::IsSymmetricPadding(paddings, data_dim);
  if (is_symmetric) {
    for (int i = 0; i < data_dim; ++i) {
      (*padding_common)[i] = paddings[2 * i];
    }
    transformed_input->ShareDataWith(input);
    transformed_input_grad->Resize(input.dims());
    return true;
  }

  *is_asymmetric_pad = true;
  std::vector<int> padding_diff(data_dim);
  std::vector<int> new_input_shape_vec(data_dim + 2);
  new_input_shape_vec[0] = input.dims()[0];
  if (data_layout != DataLayout::NHWC) {
    new_input_shape_vec[1] = input.dims()[1];
  } else {
    new_input_shape_vec[data_dim + 1] = input.dims()[data_dim + 1];
  }

  for (int i = 0; i < data_dim; ++i) {
    padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
    (*padding_common)[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
    if (data_layout != DataLayout::NHWC) {
      new_input_shape_vec[i + 2] = input.dims()[i + 2] + padding_diff[i];
      (*input_pad)[2 * i + 4] = paddings[2 * i] - (*padding_common)[i];
      (*input_pad)[2 * i + 4 + 1] = paddings[2 * i + 1] - (*padding_common)[i];
    } else {
      new_input_shape_vec[i + 1] = input.dims()[i + 1] + padding_diff[i];
      (*input_pad)[2 * i + 2] = paddings[2 * i] - (*padding_common)[i];
      (*input_pad)[2 * i + 2 + 1] = paddings[2 * i + 1] - (*padding_common)[i];
    }
  }

  transformed_input->Resize(make_ddim(new_input_shape_vec));
  dev_ctx.template Alloc<T>(transformed_input);
  transformed_input_grad->Resize(make_ddim(new_input_shape_vec));

  T pad_value(0.0);
  switch (input.dims().size()) {
    case 4:
      funcs::PadFunction<Context, T, 4>(
          dev_ctx, *input_pad, input, pad_value, transformed_input);
      break;
    case 5:
      funcs::PadFunction<Context, T, 5>(
          dev_ctx, *input_pad, input, pad_value, transformed_input);
      break;
    default:
      return false;
  }
  return true;
}

template <typename T, typename Context>
bool UseMudnnWinogradConvGradAlgorithm(const DenseTensor& input,
                                       const DenseTensor& filter,
                                       const std::vector<int>& strides,
                                       const std::vector<int>& paddings,
                                       const std::vector<int>& dilations,
                                       int groups,
                                       const std::string& data_format) {
  if (data_format != "NCHW" || input.dims().size() != 4 ||
      filter.dims().size() != 4 || strides.size() != 2 ||
      paddings.size() != 2 || dilations.size() != 2) {
    return false;
  }

  const auto in_dims = input.dims();
  const auto filter_dims = filter.dims();
  const int64_t in_channels = in_dims[1];
  const int64_t out_channels = filter_dims[0];
  const bool is_depthwise = groups == in_channels && groups == out_channels;
  return !is_depthwise && filter_dims[2] == 3 && filter_dims[3] == 3 &&
         strides[0] == 1 && strides[1] == 1 && paddings[0] == 1 &&
         paddings[1] == 1 && dilations[0] == 1 && dilations[1] == 1;
}

template <typename T, typename Context>
bool UseMudnnGemmConvGradAlgorithm(const DenseTensor& input,
                                   const DenseTensor& filter,
                                   const std::vector<int>& strides,
                                   const std::vector<int>& paddings,
                                   const std::vector<int>& dilations,
                                   int groups,
                                   const std::string& data_format) {
  if (data_format != "NCHW" || input.dims().size() != 4 ||
      filter.dims().size() != 4 || strides.size() != 2 ||
      paddings.size() != 2 || dilations.size() != 2) {
    return false;
  }

  const auto in_dims = input.dims();
  const auto filter_dims = filter.dims();
  const int64_t in_channels = in_dims[1];
  const int64_t out_channels = filter_dims[0];
  const bool is_depthwise = groups == in_channels && groups == out_channels;
  return !is_depthwise && filter_dims[2] == 1 && filter_dims[3] == 1 &&
         strides[0] == 1 && strides[1] == 1 && paddings[0] == 0 &&
         paddings[1] == 0 && dilations[0] == 1 && dilations[1] == 1;
}

template <typename T, typename Context>
::musa::dnn::Convolution::AlgorithmBwdData SelectMudnnConvBwdDataAlgorithm(
    const DenseTensor& input,
    const DenseTensor& filter,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format,
    ::musa::dnn::Convolution::AlgorithmBwdData algo) {
  if (UseMudnnWinogradConvGradAlgorithm<T, Context>(
          input, filter, strides, paddings, dilations, groups, data_format)) {
    return ::musa::dnn::Convolution::AlgorithmBwdData::WINOGRAD_NONFUSED;
  }
  if (UseMudnnGemmConvGradAlgorithm<T, Context>(
          input, filter, strides, paddings, dilations, groups, data_format)) {
    return ::musa::dnn::Convolution::AlgorithmBwdData::GEMM;
  }
  return algo;
}

template <typename T, typename Context>
::musa::dnn::Convolution::AlgorithmBwdFilter SelectMudnnConvBwdFilterAlgorithm(
    const DenseTensor& input,
    const DenseTensor& filter,
    const std::vector<int>& strides,
    const std::vector<int>& paddings,
    const std::vector<int>& dilations,
    int groups,
    const std::string& data_format,
    ::musa::dnn::Convolution::AlgorithmBwdFilter algo) {
  if (UseMudnnWinogradConvGradAlgorithm<T, Context>(
          input, filter, strides, paddings, dilations, groups, data_format)) {
    return ::musa::dnn::Convolution::AlgorithmBwdFilter::WINOGRAD_NONFUSED;
  }
  if (UseMudnnGemmConvGradAlgorithm<T, Context>(
          input, filter, strides, paddings, dilations, groups, data_format)) {
    return ::musa::dnn::Convolution::AlgorithmBwdFilter::GEMM;
  }
  return algo;
}

template <typename T, typename Context>
void MudnnConvGradKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& filter,
                         const DenseTensor& out_grad,
                         const std::vector<int>& strides,
                         const std::vector<int>& paddings_t,
                         const std::string& padding_algorithm,
                         int groups,
                         const std::vector<int>& dilations_t,
                         const std::string& data_format,
                         DenseTensor* input_grad,
                         DenseTensor* filter_grad,
                         bool use_scalar_compute) {
  DenseTensor transformed_input(input.type());
  DenseTensor transformed_input_grad(input.type());
  std::vector<int> padding_common;
  std::vector<int> dilations;
  std::vector<int> input_pad;
  bool is_asymmetric_pad = false;
  if (!PrepareMudnnConvGradInput<T, Context>(dev_ctx,
                                             input,
                                             filter,
                                             strides,
                                             paddings_t,
                                             padding_algorithm,
                                             dilations_t,
                                             data_format,
                                             &transformed_input,
                                             &transformed_input_grad,
                                             &padding_common,
                                             &dilations,
                                             &input_pad,
                                             &is_asymmetric_pad)) {
    ConvGradKernel<T>(dev_ctx,
                      input,
                      filter,
                      out_grad,
                      strides,
                      paddings_t,
                      padding_algorithm,
                      dilations_t,
                      groups,
                      data_format,
                      input_grad,
                      filter_grad);
    return;
  }

  if (IsBf16<T>()) {
    ConvGradKernel<T>(dev_ctx,
                      input,
                      filter,
                      out_grad,
                      strides,
                      paddings_t,
                      padding_algorithm,
                      dilations_t,
                      groups,
                      data_format,
                      input_grad,
                      filter_grad);
    return;
  }

  auto& h = GetMudnnHandle<Context>(dev_ctx);
  auto x_mt = CreateMUTensor(transformed_input);
  auto w_mt = CreateMUTensor(filter);
  auto dout_mt = CreateMUTensor(out_grad);

  ::musa::dnn::Convolution conv;
  if (use_scalar_compute) {
    const auto compute_mode = std::is_same<T, phi::float16>::value
                                  ? ::musa::dnn::Convolution::ComputeMode::TENSOR
                                  : ::musa::dnn::Convolution::ComputeMode::SCALAR;
    MUDNN_CHECK(conv.SetComputeMode(compute_mode), "SetComputeMode");
  }
  MUDNN_CHECK(conv.SetGroups(groups), "SetGroups");
  MUDNN_CHECK(conv.SetNdInfo(static_cast<int>(padding_common.size()),
                             padding_common.data(),
                             strides.data(),
                             dilations.data()),
              "SetNdInfo");

  auto place = dev_ctx.GetPlace();
  ::musa::dnn::MemoryMaintainer maintainer =
      [place](size_t bytes) { return PaddleInternalMemAlloc(bytes, place); };

  if (input_grad) {
    if (is_asymmetric_pad) {
      dev_ctx.template Alloc<T>(&transformed_input_grad);
    } else {
      dev_ctx.template Alloc<T>(input_grad);
      transformed_input_grad.ShareDataWith(*input_grad);
    }
    auto dx_mt = CreateMUTensor(transformed_input_grad);
    ::musa::dnn::Convolution::AlgorithmBwdData algo;
    MUDNN_CHECK(conv.GetRecommendBackwardDataAlgorithm(
                    h, algo, dx_mt, dout_mt, w_mt),
                "GetRecommendBackwardDataAlgorithm");
    algo = SelectMudnnConvBwdDataAlgorithm<T, Context>(transformed_input,
                                                       filter,
                                                       strides,
                                                       padding_common,
                                                       dilations,
                                                       groups,
                                                       data_format,
                                                       algo);
    MUDNN_CHECK(conv.RunBwdData(h, dx_mt, dout_mt, w_mt, algo, maintainer),
                "Run Mudnn DepthwiseConv BwdData");

    if (is_asymmetric_pad) {
      dev_ctx.template Alloc<T>(input_grad);
      std::vector<int> starts(input.dims().size(), 0);
      std::vector<int> axes(input.dims().size(), 0);
      for (size_t i = 0; i < input.dims().size(); ++i) {
        starts[i] = input_pad[2 * i];
        axes[i] = i;
      }
      if (input.dims().size() == 4) {
        RemovePaddingSlice<Context, T, 4>(
            dev_ctx, &transformed_input_grad, input_grad, starts, axes);
      } else {
        RemovePaddingSlice<Context, T, 5>(
            dev_ctx, &transformed_input_grad, input_grad, starts, axes);
      }
    }
  }

  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    auto dw_mt = CreateMUTensor(*filter_grad);
    ::musa::dnn::Convolution::AlgorithmBwdFilter algo;
    MUDNN_CHECK(conv.GetRecommendBackwardFilterAlgorithm(
                    h, algo, dw_mt, x_mt, dout_mt),
                "GetRecommendBackwardFilterAlgorithm");
    algo = SelectMudnnConvBwdFilterAlgorithm<T, Context>(transformed_input,
                                                         filter,
                                                         strides,
                                                         padding_common,
                                                         dilations,
                                                         groups,
                                                         data_format,
                                                         algo);
    MUDNN_CHECK(conv.RunBwdFilter(h, dw_mt, x_mt, dout_mt, algo, maintainer),
                "Run Mudnn DepthwiseConv BwdFilter");
  }
}

template <typename T, typename Context>
void DepthwiseConvGradKernel(const Context& dev_ctx,
                             const DenseTensor& input,
                             const DenseTensor& filter,
                             const DenseTensor& out_grad,
                             const std::vector<int>& strides,
                             const std::vector<int>& paddings_t,
                             const std::string& padding_algorithm,
                             int groups,
                             const std::vector<int>& dilations_t,
                             const std::string& data_format,
                             DenseTensor* input_grad,
                             DenseTensor* filter_grad) {
  MudnnConvGradKernel<T>(dev_ctx,
                         input,
                         filter,
                         out_grad,
                         strides,
                         paddings_t,
                         padding_algorithm,
                         groups,
                         dilations_t,
                         data_format,
                         input_grad,
                         filter_grad,
                         false);
}

template <typename T, typename Context>
void Conv2DGradKernel(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& filter,
                      const DenseTensor& out_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings_t,
                      const std::string& padding_algorithm,
                      const std::vector<int>& dilations_t,
                      int groups,
                      const std::string& data_format,
                      DenseTensor* input_grad,
                      DenseTensor* filter_grad) {
  MudnnConvGradKernel<T>(dev_ctx,
                         input,
                         filter,
                         out_grad,
                         strides,
                         paddings_t,
                         padding_algorithm,
                         groups,
                         dilations_t,
                         data_format,
                         input_grad,
                         filter_grad,
                         true);
}

template <typename T, typename Context>
void SlowConvDilatedGradKernel(const Context& dev_ctx,
                               const DenseTensor& input,
                               const DenseTensor& filter,
                               const paddle::optional<DenseTensor>& bias,
                               const DenseTensor& output_grad,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::string& padding_algorithm,
                               const std::vector<int>& dilations,
                               int groups,
                               const std::string& data_format,
                               DenseTensor* input_grad,
                               DenseTensor* filter_grad,
                               DenseTensor* bias_grad) {
  SlowConvBackward<T, Context, 2>(dev_ctx,
                                  input,
                                  filter,
                                  output_grad,
                                  strides,
                                  paddings,
                                  padding_algorithm,
                                  dilations,
                                  groups,
                                  data_format,
                                  input_grad,
                                  filter_grad,
                                  bias_grad);
}

template <typename T, typename Context>
void SlowConv3DDilatedGradKernel(const Context& dev_ctx,
                                 const DenseTensor& input,
                                 const DenseTensor& filter,
                                 const paddle::optional<DenseTensor>& bias,
                                 const DenseTensor& out_grad,
                                 const std::vector<int>& strides,
                                 const std::vector<int>& paddings,
                                 const std::string& padding_algorithm,
                                 int groups,
                                 const std::vector<int>& dilations,
                                 const std::string& data_format,
                                 DenseTensor* input_grad,
                                 DenseTensor* filter_grad,
                                 DenseTensor* bias_grad) {
  SlowConvBackward<T, Context, 3>(dev_ctx,
                                  input,
                                  filter,
                                  out_grad,
                                  strides,
                                  paddings,
                                  padding_algorithm,
                                  dilations,
                                  groups,
                                  data_format,
                                  input_grad,
                                  filter_grad,
                                  bias_grad);
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(conv2d_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::Conv2DGradKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(
    depthwise_conv2d_grad, musa, ALL_LAYOUT, phi::DepthwiseConvGradKernel, float, double) {}

PD_CUSTOM_KERNEL_REGISTER(
    conv3d_grad, musa, ALL_LAYOUT, phi::Conv3DGradKernel, float, double, phi::float16, phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(slow_conv2d_dilated_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::SlowConvDilatedGradKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(slow_conv3d_dilated_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::SlowConv3DDilatedGradKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(conv2d_double_grad,
                          musa,
                          ALL_LAYOUT,
                          phi::ConvGradGradKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}
