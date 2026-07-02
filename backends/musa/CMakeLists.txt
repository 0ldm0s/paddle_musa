# Copyright (c) 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All
# rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License

cmake_minimum_required(VERSION 3.10)

project(paddle-musa CXX C)

set(MUSA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(MUSA_IN_PADDLE_SUB_BUILD ON)
else()
  set(MUSA_IN_PADDLE_SUB_BUILD OFF)
endif()

if(CUSTOM_DEVICE_CMAKE_ARGS)
  foreach(arg IN LISTS CUSTOM_DEVICE_CMAKE_ARGS)
    if(arg MATCHES "^-D([^=]+)=(.*)$")
      set(${CMAKE_MATCH_1}
          "${CMAKE_MATCH_2}"
          CACHE STRING "" FORCE)
    elseif(arg MATCHES "^-D([^=]+)$")
      set(${CMAKE_MATCH_1}
          ON
          CACHE BOOL "" FORCE)
    endif()
  endforeach()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${MUSA_SOURCE_DIR}/cmake")

set(MUSA_PATH
    "/usr/local/musa"
    CACHE PATH "Path to which musa has been installed")
list(APPEND CMAKE_MODULE_PATH "${MUSA_PATH}/cmake")
find_package(MUSA REQUIRED)

set(PLUGIN_NAME "paddle-musa")
set(PLUGIN_VERSION "1.0.0")
if(NOT DEFINED PADDLE_SOURCE_DIR)
  get_filename_component(PADDLE_SOURCE_DIR "${MUSA_SOURCE_DIR}/../../Paddle" ABSOLUTE)
endif()
set(THIRD_PARTY_PATH
    "${PADDLE_SOURCE_DIR}/build/third_party"
    CACHE PATH "Third party libraries directory.")

option(WITH_TESTING "compile with unit testing" ON)
option(ON_INFER "compile with inference c++ lib" OFF)
option(WITH_MKL "compile with mkl support" ON)
option(WITH_COVERAGE "Compile PaddlePaddle with code coverage" OFF)
option(WITH_MUSA "Compile PaddlePaddle with MUSA" ON)
option(WITH_MCCL "Compile PaddlePaddle with MCCL support" ON)
set(WITH_MKLML ON)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "AR tools: ${CMAKE_AR}")

if(NOT WITH_TESTING)
  set(NO_PADDLE_SUBMODULE ON)
endif()

if(NOT WITH_MUSA AND WITH_MCCL)
  message(
    WARNING "Disable MCCL when compiling without MUSA. Force WITH_MCCL=OFF.")
  set(WITH_MCCL
      OFF
      CACHE STRING "Disable MCCL when compiling without MUSA" FORCE)
endif()

# build shared library
include(utils)
if(MUSA_IN_PADDLE_SUB_BUILD)
  set(PADDLE_INC_DIR "${PADDLE_SOURCE_DIR}")
else()
  include(paddle)
  include(external/eigen)
  include(version)
  include(generic)
  include(cblas)
  include(external/warpctc)
  include(external/warprnnt)
endif()

list(APPEND CMAKE_MODULE_PATH "${PADDLE_SOURCE_DIR}/cmake"
     "${PADDLE_SOURCE_DIR}/cmake/external")

#include_directories(${MUSA_SOURCE_DIR})
include_directories(
  ${PADDLE_INC_DIR}
  ${MUSA_SOURCE_DIR}
  ${MUSA_SOURCE_DIR}/kernels
  ${MUSA_SOURCE_DIR}/kernels/paddle_impl
  ${MUSA_SOURCE_DIR}/kernels/paddle_dnn
  ${MUSA_SOURCE_DIR}/kernels/paddle_kernels
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/blas
  ${MUSA_SOURCE_DIR}/kernels/flash_attn
  ${MUSA_SOURCE_DIR}/hack/cuda_hack
  ${MUSA_SOURCE_DIR}/hack/thrust_hack
  ${MUSA_SOURCE_DIR}/hack/paddle_backend_dyload_hack
  ${MUSA_SOURCE_DIR}/hack/paddle_platform_dyload_hack)
file(
  GLOB
  PADDLE_KERNEL_SRCS
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unique_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/warpctc_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/top_p_sampling_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/fp8_gemm_blockwise_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/warpctc_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_combine_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/cal_aux_loss_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/yolo_box_post_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/int_bincount.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_combine_no_weight_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_permute_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_combine_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/cal_aux_loss_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/slice_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/ext_build_src_rank_and_local_expert_id_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_combine_no_weight_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/fp8_quant_blockwise_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/elementwise_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/mask_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/triangular_solve_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/expand_modality_expert_id_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/triangular_solve_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_permute_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sparse_utils_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_ops_partial_nosoftmaxtopk_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/legacy_generate_proposals_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/weighted_sample_neighbors_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_and_quant_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/transpose_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_ops_partial_nosoftmaxtopk_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/moe_gate_dispatch_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/coalesce_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/top_k_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/top_k_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/poisson_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/save_combine_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/save_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/save_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/svd_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/qr_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/prune_gate_by_capacity_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/prior_box_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/random_routing_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fused_rms_norm_quant_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rprop_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/slogdeterminant_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_u_recv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/shard_index_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/spectral_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rank_attention_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/spectral_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rank_attention_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/repeat_interleave_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/repeat_interleave_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_u_recv_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/stft_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/strided_elementwise_copy_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/solve_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_ue_recv_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_uv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sigmoid_cross_entropy_with_logits_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_uv_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/send_ue_recv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sigmoid_cross_entropy_with_logits_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/instance_norm_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/instance_norm_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/layer_norm_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/load_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/load_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multinomial_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/layer_norm_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/load_combine_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/flatten2_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/moe_permute_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/flatten2_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/isclose_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nll_loss_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/moe_unpermute_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/grid_sample_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lu_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nanmedian_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/strings/gpu/strings_copy_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lu_solve_kernle.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/matrix_rank_tol_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/embedding_with_scaled_gradient_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lu_solve_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/logsumexp_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nanmedian_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/frobenius_norm_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/kthvalue_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/kthvalue_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lstsq_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/masked_select_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multiclass_nms3_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fused_token_prune_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/masked_select_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/kron_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/elementwise_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/logsumexp_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/frobenius_norm_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_scale_bias_relu_conv_bn_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_dropout_add_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_dropout_add_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_dconv_drelu_dbn_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_attention_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/fused_attention_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/max_pool2d_v2_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_layernorm_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_scale_bias_add_relu_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_multi_transformer_int8_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/class_center_sample_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/max_pool2d_v2_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_attention_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/fused_attention_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_multi_transformer_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/binomial_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dgc_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/conv_transpose_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/conv_transpose_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/depthwise_conv_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/depthwise_conv_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_dot_product_attention_op.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/skip_layernorm_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/check_numerics_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/assign_pos_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/affine_grid_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/masked_multihead_attention_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_weighted_swiglu_act_quant_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/resnet_unit_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_softmax_mask_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_softmax_mask_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_embedding_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cudnn_lstm_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_embedding_eltwise_layernorm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dist_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cholesky_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_transpose_wlch_split_quant_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/multihead_matmul_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/decode_jpeg_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_stack_transpose_quant_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cholesky_solve_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_act_dequant_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_swiglu_weighted_bwd_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_seqpool_cvm_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/apply_per_channel_scale_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/asgd_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_seqpool_cvm_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/resnet_unit_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cuda_gemm_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_softmax_mask_upper_triangle_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/strings/gpu/strings_lower_upper_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/qkv_unpack_mha_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/accuracy_check_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_transpose_split_quant_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/affine_grid_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_softmax_mask_upper_triangle_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cholesky_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cholesky_solve_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/determinant_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_feedforward_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bias_dropout_residual_layer_norm_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/block_multi_head_attention_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/blha_get_max_len.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/add_n_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_gate_attention_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bias_dropout_residual_layer_norm_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_feedforward_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/grid_sample_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cudnn_lstm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/distributed_fused_lamb_init_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gammaincc_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gammaincc_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_embedding_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/flash_attn_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/warprnnt_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/warprnnt_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/batch_norm_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/batch_norm_grad_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/softmax.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/softmax_grad.cu
  #---------------------------------------------------------------------------------------------------
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gelu_kernel.cu
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gelu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/cpu/graph_khop_sampler_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/batch_fc_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/batch_fc_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bias_act_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dropout_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dropout_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/npu_identity_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/strided_slice_kernel.cc
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_entropy_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/kps/reduce_max_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/unsqueeze_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/unsqueeze_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/squeeze_grad_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/squeeze_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/trace_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/randint_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/randint_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gaussian_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gumbel_softmax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/stack_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/stack_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_elementwise_put_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/p_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/elementwise_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/p_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_elementwise_put_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/reduce_as_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/slice_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/empty_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sign_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_sum_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_mean_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_all_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/assign_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/compare_kernel.cu            
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/kps/compare_kernel.cu     
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/elementwise_kernel.cu        
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/kps/elementwise_kernel.cu 
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/set_value_kernel.cu          
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pad_kernel.cu                
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scale_kernel.cu              
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/reduce_kernel.cu             
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/reduce_kernel.cu             
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_kernel.cu               
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nonzero_kernel.cu            
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/full_kernel.cu               
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/add_n_kernel.cu              
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cast_kernel.cu               
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_elemwise_activation_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bn_add_activation_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_elemwise_activation_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fusion_transpose_flatten_concat_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/diagonal_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_gate_attention_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/distribute_fpn_proposals_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/add_n_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bn_add_activation_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/limit_by_capacity_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/mv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/mv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_tree_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multi_dot_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/number_count_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multi_dot_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/prod_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/exponential_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/squared_l2_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/prelu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rrelu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/standard_gamma_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/renorm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sparse_momentum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/renorm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/conv_kernel_igemm.cu #todo
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/anchor_generator_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unstack_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unpool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unpool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sync_batch_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/uniform_random_batch_size_like_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sync_batch_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sync_batch_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/uniform_inplace_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/uniform_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/uniform_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unstack_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/lookup_table_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/label_smooth_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lookup_table_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/lookup_table_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/kron_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/expand_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/full_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/matmul_grad_kernel.cu #todo
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/expand_as_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lamb_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/lamb_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lerp_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mean_all_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/margin_cross_entropy_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/generate_proposals_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/margin_cross_entropy_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/histogram_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lookup_table_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lars_momentum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/interpolate_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/inverse_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/matrix_power_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lstm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/matrix_power_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/one_hot_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/one_hot_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lstm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bilinear_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/accuracy_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cumprod_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/complex_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bilinear_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/viterbi_decode_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rrelu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_nd_add_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/scatter_nd_add_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bmm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bmm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/baddbmm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cvm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cvm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_nd_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gather_nd_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/log_softmax_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/log_softmax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/conv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/conv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/deformable_conv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/deformable_conv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/conv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/conv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/conv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/conv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/baddbmm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/addmm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/addmm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/addmm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/addmm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/pool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_fc_elementwise_layernorm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gru_unit_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gru_unit_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/softmax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/softmax_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/transpose_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/diagonal_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dirichlet_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/activation_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/angle_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sync_batch_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/uniform_random_batch_size_like_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/broadcast_tensors_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_entropy2_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_by_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/clip_by_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bce_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/abs_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/abs_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adamw_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/adamw_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adamax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/angle_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/activation_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adadelta_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/adagrad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/affine_channel_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/affine_channel_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/amp_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/angle_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/ap_facade_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/ap_trivial_fusion_begin_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/ap_trivial_fusion_end_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/ap_variadic_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/arange_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/arg_min_max_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/argsort_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/as_complex_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/as_real_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/atan2_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/atan2_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/average_accumulates_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/barrier_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bce_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/beam_search_decode_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/beam_search_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/bincount_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/box_clip_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/box_coder_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/broadcast_tensors_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_identity_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_softmax_with_cross_entropy_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_softmax_with_cross_entropy_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_softmax_with_multi_label_cross_entropy_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_softmax_with_multi_label_cross_entropy_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/c_split_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/calc_reduced_attn_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/channel_shuffle_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/channel_shuffle_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/clip_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/comm_init_all_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/complex_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/concat_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/concat_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/contiguous_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/conv_test_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/correlation_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/correlation_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/crop_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/crop_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cross_entropy2_grad_kernel.cu
  # ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/ctc_align_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cum_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cum_maxmin_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/cum_maxmin_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/debug_tools_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/decayed_adagrad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/depend_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dequantize_abs_max_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dequantize_log_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/determinant_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dgc_clip_by_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dgc_momentum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/diag_embed_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/diag_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/diag_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/digamma_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/digamma_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dist_concat_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dot_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/dot_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/edit_distance_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/eigh_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/eigh_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/eigvalsh_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/eigvalsh_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/einsum_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/einsum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/erf_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/erf_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/erfinv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/erfinv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/expand_as_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/expand_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/eye_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fake_dequantize_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fake_quantize_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fetch_barrier_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fft_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fft_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_diagonal_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_diagonal_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_diagonal_tensor_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_diagonal_tensor_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fill_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/flash_attn_v3_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/flash_attn_v3_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/flash_attn_v3_utils.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/flip_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fold_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fold_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/frame_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/frame_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/ftrl_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/fused_adam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gammaln_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gammaln_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gaussian_inplace_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/global_gather_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/global_scatter_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/group_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gru_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gru_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/gumbel_softmax_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/hinge_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/hinge_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/huber_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/huber_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i0_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i0_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i0e_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i0e_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i1_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i1_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i1e_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/i1e_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/im2sequence_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/im2sequence_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/increment_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_add_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_elementwise_get_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_elementwise_get_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_put_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_put_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_sample_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_sample_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_select_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/index_select_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/interpolate_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/inverse_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/kldiv_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/kldiv_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/l1_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/l1_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/label_smooth_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lerp_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lgamma_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lgamma_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/linspace_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/llm_int8_linear_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lod_reset_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lod_reset_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/log_loss_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/log_loss_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/logcumsumexp_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/logspace_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lrn_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lrn_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lu_unpack_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/lu_unpack_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/masked_fill_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/masked_fill_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/matrix_rank_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/maxout_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/maxout_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mean_all_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/merged_momentum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mode_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mode_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/momentum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/moving_average_abs_max_scale_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/mp_allreduce_sum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multiplex_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/multiplex_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nadam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nms_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/nop_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/numel_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/overlap_add_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/overlap_add_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pad3d_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pad3d_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pad_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/partial_allgather_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/partial_concat_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/partial_concat_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/partial_recv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/partial_send_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/partial_sum_kernel.cu # empty registry
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pixel_shuffle_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pixel_shuffle_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pixel_unshuffle_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pixel_unshuffle_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/poisson_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/polygamma_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/polygamma_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/pow2_decay_with_linear_warmup_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/prelu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/psroi_pool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/psroi_pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/put_along_axis_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/put_along_axis_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/qr_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/quant_linear_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/quantize_linear_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/radam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/range_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/repeat_interleave_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/reduce_as_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/rmsprop_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/roi_align_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/roi_align_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/roi_pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/roll_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/roll_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/row_conv_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/row_conv_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/searchsorted_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/seed_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/segment_pool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/segment_pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/selu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_expand_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_expand_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_mask_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_pool_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_pool_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_softmax_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sequence_softmax_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/set_value_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/share_data_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/shuffle_channel_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/shuffle_channel_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/slogdeterminant_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/soft_relu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/soft_relu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/solve_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/split_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/squared_l2_norm_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/stft_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/straight_through_estimator_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/strided_slice_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/strided_slice_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/svd_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/swiglu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/swiglu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/sync_calc_stream_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/take_along_axis_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/take_along_axis_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/temporal_shift_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/temporal_shift_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tile_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tile_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/trace_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/transpose_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tril_indices_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tril_triu_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/tril_triu_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/triu_indices_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/trunc_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/trunc_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/truncated_gaussian_random_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unbind_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unfold_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unfold_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/uniform_inplace_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/unique_consecutive_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/weight_dequantize_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/weight_only_linear_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/where_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/where_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/yolo_box_head_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/yolo_box_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fc_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bn_activation_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_bn_activation_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_conv2d_add_act_kernel.cu #todo
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_gemm_epilogue_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_gemm_epilogue_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_linear_param_grad_add_kernel.cu #todo
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_partial_rope_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_partial_rope_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_rope_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fused_rope_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/fusion/gpu/fusion_group_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/legacy_crop_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/legacy_crop_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/legacy_expand_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/legacy/gpu/legacy_expand_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/adam_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/clip_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/dgc_clip_by_norm_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/ftrl_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/get_tensor_from_selected_rows_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/selected_rows/gpu/share_data_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/mask_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/reshape_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/reshape_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/slice_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sparse_attention_kernel.cu #todo
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sum_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/sum_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/transpose_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/unary_grad_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/sparse/gpu/unary_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/bitwise_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/kps/logical_kernel.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/affine_grid_utils.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/math_function.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/cross_entropy.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/pooling.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/concat_and_split_functor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/sequence*.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/matrix_reduce.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/deformable_conv_functor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/fake_dequantize_functor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/gather_scatter_functor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/gru_compute.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/im2col.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/lstm_compute.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/maxouting.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/radix_sort.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/segment_pooling.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/selected_rows_functor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/skip_layernorm_functor.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/vol2col.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/broadcast.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/pad.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/slice.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/sequence_pooling.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/l1_norm.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/elementwise.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/constant.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/loss.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/sequence2batch.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/erf.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/funcs/eigen/sign.cu
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_amax_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_amin_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_any_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reduce_min_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/prod_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/reverse_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/shape_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/is_empty_kernel.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/meshgrid_kernel.cu.cc
  ${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/meshgrid_grad_kernel.cu.cc
  #${PADDLE_SOURCE_DIR}/paddle/phi/kernels/gpu/top_k_grad_kernel.cu
  #in paddle_musa 
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/softmax.mu
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/emb_eltwise_layer_norm_functor.cu
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/fake_quantize_functor.mu
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/fft.mu
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/matrix_inverse.mu
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/matrix_solve.mu
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/multihead_matmul_functor.mu
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/repeat_tensor2index_tensor.mu
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/weight_only_gemv.mu
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/fc_functor.mu
  ${MUSA_SOURCE_DIR}/kernels/paddle_funcs/math_function_extra.mu

  )

file(
  GLOB_RECURSE PLUGIN_SRCS
  RELATIVE ${MUSA_SOURCE_DIR}
  ${MUSA_SOURCE_DIR}/runtime/*.cc 
  runtime/*.mu 
  kernels/musa*.cc
  kernels/paddle_kernels/*.mu
  kernels/paddle_kernels/*.cc
  kernels/flash_attn/*.mu
  hack/paddle_backend_dyload_hack/*.cc 
  hack/paddle_platform_dyload_hack/*.cc 
  hack/paddle_backend_dyload_hack/*.mu)
set(CUSTOM_DEVICE_SRCS ${PLUGIN_SRCS} ${PADDLE_KERNEL_SRCS})

include(musa)
include(mudnn)
musa_add_library(${PLUGIN_NAME} SHARED ${CUSTOM_DEVICE_SRCS})

target_include_directories(
  ${PLUGIN_NAME} PUBLIC 
          ${PADDLE_SOURCE_DIR}
	  ${MUSA_SOURCE_DIR}
          ${MUSA_SOURCE_DIR}/kernels
          ${MUSA_SOURCE_DIR}/hack/cuda_hack
          ${MUSA_SOURCE_DIR}/hack/thrust_hack
	  ${PADDLE_SOURCE_DIR}/third_party/pybind/include
          ${PADDLE_SOURCE_DIR}/paddle/phi/api/include/compat)
 #link_directories(${PADDLE_LIB_DIR})

#if(ON_INFER)
#  target_link_directories(${PLUGIN_NAME} ${PADDLE_INFERENCE_LIB_DIR})
#  target_link_libraries(${PLUGIN_NAME} paddle_inference)
#endif()

# link musa rt mudnn paddle
if(MUSA_IN_PADDLE_SUB_BUILD)
  set(MUSA_PADDLE_LINK_LIBS)
  if(TARGET paddle)
    list(APPEND MUSA_PADDLE_LINK_LIBS paddle)
  endif()
  if(TARGET phi_core)
    list(APPEND MUSA_PADDLE_LINK_LIBS phi_core)
  endif()
  if(TARGET phi_gpu)
    list(APPEND MUSA_PADDLE_LINK_LIBS phi_gpu)
  endif()
  message(STATUS "MUSA_PADDLE_LINK_LIBS: ${MUSA_PADDLE_LINK_LIBS}")
  target_link_libraries(
    ${PLUGIN_NAME}
    ${MUSA_PADDLE_LINK_LIBS}
    ${musart_lib}
    ${mudnn_lib})
else()
  get_filename_component(PADDLE_BASE_LIB_DIR ${PADDLE_CORE_LIB} DIRECTORY)
  get_filename_component(PADDLE_PACKAGE_DIR ${PADDLE_BASE_LIB_DIR} DIRECTORY)
  set(PADDLE_PHI_CORE_LIB ${PADDLE_PACKAGE_DIR}/libs/libphi_core.so)
  message(STATUS "PADDLE_PHI_CORE_LIB: ${PADDLE_PHI_CORE_LIB}")
  target_link_libraries(
    ${PLUGIN_NAME}
    ${PADDLE_CORE_LIB}
    "-Wl,--no-as-needed"
    ${PADDLE_PHI_CORE_LIB}
    "-Wl,--as-needed"
    ${musart_lib}
    ${mudnn_lib})
  target_link_options(
    ${PLUGIN_NAME}
    PRIVATE
    "-Wl,-rpath,${PADDLE_BASE_LIB_DIR}"
    "-Wl,-rpath,${PADDLE_PACKAGE_DIR}/libs"
    "-Wl,-rpath,\$ORIGIN/../paddle/base"
    "-Wl,-rpath,\$ORIGIN/../paddle/libs")
endif()

# link mccl
if(WITH_MCCL)
  add_definitions("-DPADDLE_WITH_MCCL")
  include(mccl)
  target_link_libraries(${PLUGIN_NAME} ${mccl_lib})
else()
  if(WITH_MUSA)
    message(
      WARNING
        "If the environment is multi-card, the WITH_MCCL option needs to be turned on, otherwise only a single card can be used."
    )
  endif()
endif()

# link third party
if(MUSA_IN_PADDLE_SUB_BUILD)
  target_link_libraries(
    ${PLUGIN_NAME}
    glog
    eigen3
    gflags
    warpctc
    warprnnt)
else()
  include(third_party)
  add_dependencies(${PLUGIN_NAME} third_party)
  target_link_libraries(
    ${PLUGIN_NAME}
    glog
    eigen3
    gflags
    ${WARPCTC_LIBRARIES}
    ${WARPRNNT_LIBRARIES})
endif()

include_directories(BEFORE ${PADDLE_SOURCE_DIR})

target_compile_definitions(
  ${PLUGIN_NAME}
  PUBLIC __NVCC__=__MUSACC__
         __CUDACC_VER_MAJOR__=99
         __CUDACC_VER_MINOR__=10000
         MARCH_TYPE=310 # TODO: the version num should be obtained automatically
                        # here
			#CUDNN_VERSION=100100 # 先全部放开，再找mudnn对标（2025.10.27）
	 GPUContext=CustomContext
         KPSContext=CustomContext
         STREAM_TYPE=musaStream_t
         EVENT_TYPE=musaEvent_t
         EIGEN_USE_GPU=1
         PADDLE_WITH_MUSA=1
         PADDLE_WITH_MCCL=1
         PADDLE_USE_MUBLAS=1
         PADDLE_WITH_CUSTOM_DEVICE=1
         EIGEN_USE_MUSA=1)

find_package(
  Python
  COMPONENTS Interpreter
  REQUIRED)

if(MUSA_IN_PADDLE_SUB_BUILD)
  set(MUSA_PADDLE_CUSTOM_DEVICE_DIR
      "${PADDLE_BINARY_DIR}/python/paddle/paddle_custom_device")
  add_custom_command(
    OUTPUT ${MUSA_PADDLE_CUSTOM_DEVICE_DIR}/.timestamp
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${MUSA_PADDLE_CUSTOM_DEVICE_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${MUSA_PADDLE_CUSTOM_DEVICE_DIR}/include
    COMMAND ${CMAKE_COMMAND} -E touch
            ${MUSA_PADDLE_CUSTOM_DEVICE_DIR}/__init__.py
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${PLUGIN_NAME}>
            ${MUSA_PADDLE_CUSTOM_DEVICE_DIR}/
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${MUSA_SOURCE_DIR}/runtime
            ${MUSA_PADDLE_CUSTOM_DEVICE_DIR}/include/runtime
    COMMAND ${CMAKE_COMMAND} -E touch
            ${MUSA_PADDLE_CUSTOM_DEVICE_DIR}/.timestamp
    DEPENDS ${PLUGIN_NAME}
    COMMENT "Copy MUSA custom device files to Paddle python package")
  add_custom_target(custom_device_package_copy
                    DEPENDS ${MUSA_PADDLE_CUSTOM_DEVICE_DIR}/.timestamp)
else()
  # packing wheel package
  configure_file(${MUSA_SOURCE_DIR}/setup.py.in
                 ${CMAKE_CURRENT_BINARY_DIR}/setup.py)

  add_custom_command(
    TARGET ${PLUGIN_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/python/
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/lib${PLUGIN_NAME}.so
      ${CMAKE_CURRENT_BINARY_DIR}/python/paddle_custom_device/
    COMMENT "Creating plugin directories------>>>")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/setup.py bdist_wheel
    DEPENDS ${PLUGIN_NAME}
    COMMENT "Packing whl packages------>>>")

  add_custom_target(python_package ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/python/.timestamp)
endif()

if(WITH_TESTING)
  set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Paddle")
  enable_testing()
  add_subdirectory(tests)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp
    COMMAND cp -r ${MUSA_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR})
  add_custom_target(python_tests ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/tests/.timestamp)
endif()
