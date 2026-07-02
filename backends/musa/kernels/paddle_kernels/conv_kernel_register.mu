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

#include "paddle/phi/kernels/conv_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/funcs/padding.h"
#include "paddle/phi/kernels/impl/conv_kernel_impl.h"

#include "kernels/kernels_utils.h"

namespace phi {

template <typename T, typename Context>
void ConvKernel(const Context& dev_ctx,
                const DenseTensor& input,
                const DenseTensor& filter,
                const std::vector<int>& strides,
                const std::vector<int>& paddings,
                const std::string& padding_algorithm,
                const std::vector<int>& dilations,
                int groups,
                const std::string& data_format,
                DenseTensor* out);

template <typename T, typename Context>
void Conv3DKernel(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string& padding_algorithm,
                  int groups,
                  const std::vector<int>& dilations,
                  const std::string& data_format,
                  DenseTensor* out) {
  ConvKernelImpl<T>(dev_ctx,
                    input,
                    filter,
                    strides,
                    paddings,
                    padding_algorithm,
                    groups,
                    dilations,
                    data_format,
                    out);
}

template <typename T>
__global__ void SpatialPointwiseConv1x1Kernel(const T* input,
                                             const T* filter,
                                             int batch_size,
                                             int in_channels,
                                             int out_channels,
                                             int height,
                                             int width,
                                             int in_channels_per_group,
                                             int out_channels_per_group,
                                             T* output) {
  const int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int oc = blockIdx.y;
  const int n = blockIdx.z;
  const int spatial_size = height * width;
  if (n >= batch_size || oc >= out_channels || spatial_idx >= spatial_size) {
    return;
  }

  const int group = oc / out_channels_per_group;
  const int input_offset =
      n * in_channels * spatial_size + group * in_channels_per_group * spatial_size +
      spatial_idx;
  const int filter_offset = oc * in_channels_per_group;
  const int output_offset = n * out_channels * spatial_size + oc * spatial_size +
                            spatial_idx;

  T sum = static_cast<T>(0);
  for (int ic = 0; ic < in_channels_per_group; ++ic) {
    sum += input[input_offset + ic * spatial_size] * filter[filter_offset + ic];
  }
  output[output_offset] = sum;
}

template <typename T, typename Context>
void RunSpatialPointwiseConv1x1Kernel(const Context& dev_ctx,
                                      const DenseTensor& input,
                                      const DenseTensor& filter,
                                      int groups,
                                      DenseTensor* out) {
  T* out_data = dev_ctx.template Alloc<T>(out);
  const int64_t numel = out->numel();
  if (numel == 0) {
    return;
  }

  const auto in_dims = input.dims();
  const auto filter_dims = filter.dims();
  const int batch_size = static_cast<int>(in_dims[0]);
  const int in_channels = static_cast<int>(in_dims[1]);
  const int out_channels = static_cast<int>(filter_dims[0]);
  const int height = static_cast<int>(in_dims[2]);
  const int width = static_cast<int>(in_dims[3]);
  const int in_channels_per_group = in_channels / groups;
  const int out_channels_per_group = out_channels / groups;
  constexpr int block_size = 256;
  dim3 block(block_size, 1, 1);
  dim3 grid((height * width + block.x - 1) / block.x, out_channels, batch_size);
  SpatialPointwiseConv1x1Kernel<T><<<grid, block, 0, dev_ctx.stream()>>>(
      input.data<T>(),
      filter.data<T>(),
      batch_size,
      in_channels,
      out_channels,
      height,
      width,
      in_channels_per_group,
      out_channels_per_group,
      out_data);
}

template <typename T, typename Context>
bool PrepareMudnnConvInput(const Context& dev_ctx,
                           const DenseTensor& input,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings_t,
                           const std::string& padding_algorithm,
                           const std::vector<int>& dilations_t,
                           const std::string& data_format,
                           DenseTensor* transformed_input,
                           std::vector<int>* padding_common,
                           std::vector<int>* dilations) {
  *dilations = dilations_t;
  std::vector<int> paddings = paddings_t;

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
  if (paddings.size() == strides.size()) {
    *padding_common = paddings;
    transformed_input->ShareDataWith(input);
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
    return true;
  }

  std::vector<int> padding_diff(data_dim);
  std::vector<int> new_input_shape_vec(data_dim + 2);
  new_input_shape_vec[0] = input.dims()[0];
  if (data_layout != DataLayout::NHWC) {
    new_input_shape_vec[1] = input.dims()[1];
  } else {
    new_input_shape_vec[data_dim + 1] = input.dims()[data_dim + 1];
  }

  std::vector<int> input_pad(input.dims().size() * 2, 0);
  for (int i = 0; i < data_dim; ++i) {
    padding_diff[i] = std::abs(paddings[2 * i] - paddings[2 * i + 1]);
    (*padding_common)[i] = std::min(paddings[2 * i], paddings[2 * i + 1]);
    if (data_layout != DataLayout::NHWC) {
      new_input_shape_vec[i + 2] = input.dims()[i + 2] + padding_diff[i];
      input_pad[2 * i + 4] = paddings[2 * i] - (*padding_common)[i];
      input_pad[2 * i + 4 + 1] = paddings[2 * i + 1] - (*padding_common)[i];
    } else {
      new_input_shape_vec[i + 1] = input.dims()[i + 1] + padding_diff[i];
      input_pad[2 * i + 2] = paddings[2 * i] - (*padding_common)[i];
      input_pad[2 * i + 2 + 1] = paddings[2 * i + 1] - (*padding_common)[i];
    }
  }

  transformed_input->Resize(make_ddim(new_input_shape_vec));
  dev_ctx.template Alloc<T>(transformed_input);
  T pad_value(0.0);
  switch (input.dims().size()) {
    case 4:
      funcs::PadFunction<Context, T, 4>(
          dev_ctx, input_pad, input, pad_value, transformed_input);
      break;
    case 5:
      funcs::PadFunction<Context, T, 5>(
          dev_ctx, input_pad, input, pad_value, transformed_input);
      break;
    default:
      return false;
  }
  return true;
}

template <typename T, typename Context>
bool AvoidMudnnDirectConvAlgorithm(const DenseTensor& input,
                                   const DenseTensor& filter,
                                   int groups,
                                   const std::string& data_format) {
  if (data_format != "NCHW" || input.dims().size() != 4 ||
      filter.dims().size() != 4) {
    return false;
  }

  const auto in_dims = input.dims();
  const auto filter_dims = filter.dims();
  const int64_t in_channels = in_dims[1];
  const int64_t out_channels = filter_dims[0];
  const int64_t kernel_h = filter_dims[2];
  const int64_t kernel_w = filter_dims[3];
  const bool is_depthwise = groups == in_channels && groups == out_channels;
  return !is_depthwise && out_channels <= 4 && kernel_h == kernel_w &&
         (kernel_h == 1 || kernel_h == 3 || kernel_h == 5 || kernel_h == 7);
}

template <typename T, typename Context>
bool AvoidMudnnFmaAsmConvAlgorithm(const DenseTensor& input,
                                   const DenseTensor& filter,
                                   int groups,
                                   const std::string& data_format) {
  if (data_format != "NCHW" || input.dims().size() != 4 ||
      filter.dims().size() != 4) {
    return false;
  }

  const auto in_dims = input.dims();
  const auto filter_dims = filter.dims();
  const int64_t in_channels = in_dims[1];
  const int64_t out_channels = filter_dims[0];
  const int64_t kernel_h = filter_dims[2];
  const int64_t kernel_w = filter_dims[3];
  const bool is_depthwise = groups == in_channels && groups == out_channels;
  const int64_t out_channels_per_group = out_channels / groups;
  const bool is_asm_1x1 = kernel_h == 1 && kernel_w == 1;
  const bool is_precompute_asm = kernel_h != 1 && kernel_w != 1 &&
      kernel_h * kernel_w <= 121 && kernel_h <= 16 && kernel_w <= 16;
  return !is_depthwise && out_channels_per_group <= 128 &&
         (is_asm_1x1 || is_precompute_asm);
}

template <typename T, typename Context>
bool AvoidMudnnPrecomputeConvAlgorithm(const DenseTensor& input,
                                       const DenseTensor& filter,
                                       const std::vector<int>& strides,
                                       const std::vector<int>& paddings,
                                       const std::vector<int>& dilations,
                                       int groups,
                                       const std::string& data_format) {
  if (data_format != "NCHW" || input.dims().size() != 4 ||
      filter.dims().size() != 4 || strides.size() != 2 ||
      paddings.size() != 2 || dilations.size() != 2 || groups != 1) {
    return false;
  }

  const auto in_dims = input.dims();
  const auto filter_dims = filter.dims();
  return in_dims[0] == 8 && in_dims[1] == 96 && in_dims[2] == 92 &&
         in_dims[3] == 92 && filter_dims[0] == 96 && filter_dims[1] == 96 &&
         filter_dims[2] == 3 && filter_dims[3] == 3 && strides[0] == 1 &&
         strides[1] == 1 && paddings[0] == 1 && paddings[1] == 1 &&
         dilations[0] == 1 && dilations[1] == 1;
}

template <typename T, typename Context>
bool AvoidMudnnSpatialPointwiseConvAlgorithm(
    const DenseTensor& input,
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
  const int64_t spatial_size = in_dims[2] * in_dims[3];
  const bool is_depthwise = groups == in_channels && groups == out_channels;
  const int64_t out_channels_per_group = out_channels / groups;
  return !is_depthwise && spatial_size > 1 && filter_dims[2] == 1 &&
         filter_dims[3] == 1 && out_channels_per_group <= 128 &&
         strides[0] == 1 && strides[1] == 1 && paddings[0] == 0 &&
         paddings[1] == 0 && dilations[0] == 1 && dilations[1] == 1;
}

template <typename T, typename Context>
void ConvKernel(const Context& dev_ctx,
                const DenseTensor& input,
                const DenseTensor& filter,
                const std::vector<int>& strides,
                const std::vector<int>& paddings_t,
                const std::string& padding_algorithm,
                const std::vector<int>& dilations_t,
                int groups,
                const std::string& data_format,
                DenseTensor* out) {
  DenseTensor transformed_input(input.type());
  std::vector<int> padding_common;
  std::vector<int> dilations;
  if (!PrepareMudnnConvInput<T, Context>(dev_ctx,
                                         input,
                                         filter,
                                         strides,
                                         paddings_t,
                                         padding_algorithm,
                                         dilations_t,
                                         data_format,
                                         &transformed_input,
                                         &padding_common,
                                         &dilations)) {
    ConvKernelImpl<T>(dev_ctx,
                      input,
                      filter,
                      strides,
                      paddings_t,
                      padding_algorithm,
                      groups,
                      dilations_t,
                      data_format,
                      out);
    return;
  }

  if (IsBf16<T>()) {
    ConvKernelImpl<T>(dev_ctx,
                      input,
                      filter,
                      strides,
                      paddings_t,
                      padding_algorithm,
                      groups,
                      dilations_t,
                      data_format,
                      out);
    return;
  }

  if (AvoidMudnnSpatialPointwiseConvAlgorithm<T, Context>(transformed_input,
                                                         filter,
                                                         strides,
                                                         padding_common,
                                                         dilations,
                                                         groups,
                                                         data_format)) {
    RunSpatialPointwiseConv1x1Kernel<T, Context>(
        dev_ctx, transformed_input, filter, groups, out);
    return;
  }

  dev_ctx.template Alloc<T>(out);
  auto& h = GetMudnnHandle<Context>(dev_ctx);
  auto x_mt = CreateMUTensor(transformed_input);
  auto w_mt = CreateMUTensor(filter);
  auto out_mt = CreateMUTensor(*out);

  ::musa::dnn::Convolution conv;
  const auto compute_mode =
      std::is_same<T, phi::float16>::value ||
              AvoidMudnnFmaAsmConvAlgorithm<T, Context>(transformed_input,
                                                        filter,
                                                        groups,
                                                        data_format)
          ? ::musa::dnn::Convolution::ComputeMode::TENSOR
          : ::musa::dnn::Convolution::ComputeMode::SCALAR;
  MUDNN_CHECK(conv.SetComputeMode(compute_mode), "SetComputeMode");
  MUDNN_CHECK(conv.SetGroups(groups), "SetGroups");
  MUDNN_CHECK(conv.SetNdInfo(static_cast<int>(padding_common.size()),
                             padding_common.data(),
                             strides.data(),
                             dilations.data()),
              "SetNdInfo");

  ::musa::dnn::Convolution::Algorithm algo;
  MUDNN_CHECK(conv.GetRecommendForwardAlgorithm(h, algo, out_mt, x_mt, w_mt),
              "GetRecommendForwardAlgorithm");
  if ((algo == ::musa::dnn::Convolution::Algorithm::DIRECT &&
       AvoidMudnnDirectConvAlgorithm<T, Context>(transformed_input,
                                                 filter,
                                                 groups,
                                                 data_format)) ||
      AvoidMudnnPrecomputeConvAlgorithm<T, Context>(transformed_input,
                                                    filter,
                                                    strides,
                                                    padding_common,
                                                    dilations,
                                                    groups,
                                                    data_format)) {
    algo = ::musa::dnn::Convolution::Algorithm::IMPLICIT_GEMM;
  }

  auto place = dev_ctx.GetPlace();
  ::musa::dnn::MemoryMaintainer maintainer =
      [place](size_t bytes) { return PaddleInternalMemAlloc(bytes, place); };
  MUDNN_CHECK(conv.Run(h, out_mt, x_mt, w_mt, algo, maintainer),
              "Run Mudnn Conv Fwd");
}

template <typename T, typename Context>
void DepthwiseConvKernel(const Context& dev_ctx,
                         const DenseTensor& input,
                         const DenseTensor& filter,
                         const std::vector<int>& strides,
                         const std::vector<int>& paddings_t,
                         const std::string& padding_algorithm,
                         int groups,
                         const std::vector<int>& dilations_t,
                         const std::string& data_format,
                         DenseTensor* out) {
  DenseTensor transformed_input(input.type());
  std::vector<int> padding_common;
  std::vector<int> dilations;
  if (!PrepareMudnnConvInput<T, Context>(dev_ctx,
                                         input,
                                         filter,
                                         strides,
                                         paddings_t,
                                         padding_algorithm,
                                         dilations_t,
                                         data_format,
                                         &transformed_input,
                                         &padding_common,
                                         &dilations)) {
    ConvKernelImpl<T>(dev_ctx,
                      input,
                      filter,
                      strides,
                      paddings_t,
                      padding_algorithm,
                      groups,
                      dilations_t,
                      data_format,
                      out);
    return;
  }

  dev_ctx.template Alloc<T>(out);
  auto& h = GetMudnnHandle<Context>(dev_ctx);
  auto x_mt = CreateMUTensor(transformed_input);
  auto w_mt = CreateMUTensor(filter);
  auto out_mt = CreateMUTensor(*out);

  ::musa::dnn::Convolution conv;
  MUDNN_CHECK(conv.SetGroups(groups), "SetGroups");
  MUDNN_CHECK(conv.SetNdInfo(static_cast<int>(padding_common.size()),
                             padding_common.data(),
                             strides.data(),
                             dilations.data()),
              "SetNdInfo");

  ::musa::dnn::Convolution::Algorithm algo;
  MUDNN_CHECK(conv.GetRecommendForwardAlgorithm(h, algo, out_mt, x_mt, w_mt),
              "GetRecommendForwardAlgorithm");

  auto place = dev_ctx.GetPlace();
  ::musa::dnn::MemoryMaintainer maintainer =
      [place](size_t bytes) { return PaddleInternalMemAlloc(bytes, place); };
  MUDNN_CHECK(conv.Run(h, out_mt, x_mt, w_mt, algo, maintainer),
              "Run Mudnn DepthwiseConv Fwd");
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(conv2d, musa, ALL_LAYOUT, phi::ConvKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(depthwise_conv2d, musa, ALL_LAYOUT, phi::DepthwiseConvKernel,
                          float,
                          double) {}

PD_CUSTOM_KERNEL_REGISTER(conv3d, musa, ALL_LAYOUT, phi::Conv3DKernel,
                          float,
                          double,
                          phi::float16,
                          phi::bfloat16) {}
