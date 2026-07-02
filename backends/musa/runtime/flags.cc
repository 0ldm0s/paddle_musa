// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2022 NVIDIA Authors. All Rights Reserved.
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
// - [Adapt some flag to musa backend]

#include "paddle/common/flags.h"

#if defined(PADDLE_WITH_MUSA)
/**
 * CUDA related FLAG
 * Name: FLAGS_cublaslt_exhaustive_search_times
 * Since Version: 2.3.0
 * Value Range: int64_t, default=0
 * Example:
 * Note: Represents times of exhaustive search to evaluate performance of
 *       cuBlasLt matmul algorithm (with/without epilogue). Set this flag
 *       with value > 0 to enable exhaustive search. Default is 0, means
 *       getting algorithms via heuristic search. There are two search methods
 *       in cuBlasLt, heuristic search and exhaustive search. Exhaustive search
 *       attempts all cuBlasLt algorithms to select the fastest, which is very
 *       time-consuming, and the selected algorithm will be cached for a given
 *       layer specification Once you change the layer specifications
 *       (such as M, N and K), it will re-search again.
 */
PHI_DEFINE_EXPORTED_int64(
    cublaslt_exhaustive_search_times,
    0,
    "The times of exhaustive search for cuBlasLt matmul with/without "
    " epilogue algorithms, default is 0, means disabling exhaustive search.");

/**
 * CUDA related related FLAG
 * Name: FLAGS_enable_cublas_tensor_op_math
 * Since Version: 1.2.0
 * Value Range: bool, default=false
 * Example:
 * Note: whether to use Tensor Core, faster but it may loss precision.
 */
PHI_DEFINE_EXPORTED_bool(
    enable_cublas_tensor_op_math,
    false,
    "The enable_cublas_tensor_op_math indicate whether to use Tensor Core, "
    "but it may loss precision. Currently, There are two CUDA libraries that"
    " use Tensor Cores, cuBLAS and cuDNN. cuBLAS uses Tensor Cores to speed up"
    " GEMM computations(the matrices must be either half precision or single "
    "precision); cuDNN uses Tensor Cores to speed up both convolutions(the "
    "input and output must be half precision) and recurrent neural networks "
    "(RNNs).");

/**
 * CUDA related related FLAG
 * Name: FLAGS_gemm_use_half_precision_compute_type
 * Since Version: 2.4
 * Value Range: bool, default=false
 * Example:
 * Note: whether to use fp16 compute type when the input and output is fp16,
 * faster but it may loss precision.
 */
PHI_DEFINE_EXPORTED_bool(
    gemm_use_half_precision_compute_type,
    false,
    "Whether to use fp16 compute type when the input and output is fp16, "
    "faster but it may loss precision in most case. If true, the compute "
    "type will be set to fp16. Default is false.");

PHI_DEFINE_EXPORTED_bool(use_fast_math,
                         false,
                         "Whether to use fast math GPU functions.");
/**
 * CUDA related FLAG
 * Name: FLAGS_embedding_deterministic
 * Since Version: 2.5
 * Value Range: int64, default=0
 * Example:
 * Note: whether to use deterministic algorithm in embedding op.
 *       If it is 1, it will use the optimized deterministic CUDA kernel in
 *       embedding op. If it is 2, it will use the legacy deterministic
 *       CUDA kernel in embedding op.
 */
PHI_DEFINE_EXPORTED_int64(
    embedding_deterministic,
    0,
    "Whether allow using an deterministic algorithm for embedding "
    "operator. The deterministic algorithm may be slower. If "
    "it is larger than 0, the algorithm is deterministic.");

/**
 * CUDA related FLAG
 * Name: FLAGS_selected_gpus
 * Since Version: 1.3.0
 * Value Range: integer list separated by comma, default empty list
 * Example: FLAGS_selected_gpus=0,1,2,3,4,5,6,7 to train or predict with 0~7 gpu
 * cards
 * Note: A list of device ids separated by comma, like: 0,1,2,3
 */
PHI_DEFINE_EXPORTED_string(
    selected_gpus,
    "",
    "A list of device ids separated by comma, like: 0,1,2,3. "
    "This option is useful when doing multi process training and "
    "each process have only one device (GPU). If you want to use ");

/**
 * CUDNN related FLAG
 * Name: FLAGS_cudnn_deterministic
 * Since Version: 0.13.0
 * Value Range: bool, default=false
 * Example:
 * Note: whether to use deterministic algorithm in cudnn.
 *       If true, it will slow down some operators such as conv and pooling.
 */
PHI_DEFINE_EXPORTED_bool(
    cudnn_deterministic,
    false,
    "Whether allow using an autotuning algorithm for convolution "
    "operator. The autotuning algorithm may be non-deterministic. If "
    "true, the algorithm is deterministic.");


/**
 * GPU RNG related FLAG
 * Name: FLAGS_deterministic_rng
 * Since Version: 3.4
 * Value Range: bool, default=false
 * Example: paddle.set_flags({'FLAGS_deterministic_rng': True})
 * Note: Fix RNG kernel launch config so same seed gives same results
 *       across GPU types.
 */
PHI_DEFINE_EXPORTED_bool(
    deterministic_rng,
    false,
    "Enable cross-device RNG consistency by fixing GPU kernel launch "
    "configuration. When true, RNG kernels use a fixed grid/block size "
    "so that the same seed produces identical results across GPU types.");

/**
 * GPU RNG related FLAG
 * Name: FLAGS_deterministic_rng_grid
 * Since Version: 3.4
 * Value Range: int32, default=1024
 * Example: paddle.set_flags({'FLAGS_deterministic_rng_grid': 4096})
 * Note: Grid size cap used when FLAGS_deterministic_rng is enabled.
 *       Cross-device consistency requires the same value on all devices.
 */
PHI_DEFINE_EXPORTED_int32(
    deterministic_rng_grid,
    1024,
    "Grid size cap when FLAGS_deterministic_rng is enabled.");

#endif
