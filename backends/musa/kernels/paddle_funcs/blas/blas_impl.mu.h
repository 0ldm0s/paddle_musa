//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
#if defined(__MUSACC__)
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#endif
#include "glog/logging.h"
#include "paddle/common/flags.h"

#include "mublas_dynload.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#define INT_MAX_VALUE 2147483647

COMMON_DECLARE_bool(enable_cublas_tensor_op_math);
COMMON_DECLARE_bool(gemm_use_half_precision_compute_type);

namespace phi {
namespace funcs {

template <typename T>
struct CUBlas;

template <>
struct CUBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSaxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasScopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgemmBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "SgemmBatched is not supported on musa mp31"));
#endif
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::mublasSgemmStridedBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "SgemmStridedBatched is not supported on musa mp31"));
#endif
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/mublas/index.html#mublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      mublasOperation_t transa,
                      mublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const float *alpha,
                      const void *A,
                      musaDataType_t Atype,
                      int lda,
                      const void *B,
                      musaDataType_t Btype,
                      int ldb,
                      const float *beta,
                      void *C,
                      musaDataType_t Ctype,
                      int ldc) {
// Because the gcc 4.8 doesn't expand template parameter pack that
// appears in a lambda-expression, I can not use template parameter pack
// here.
#if !defined(PADDLE_WITH_MUSA)
    VLOG(5) << "use_tensor_op_math: "
            << (dev_ctx->tensor_core_available() ? "True" : "False");
    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgemmEx(handle,
                                                             transa,
                                                             transb,
                                                             m,
                                                             n,
                                                             k,
                                                             alpha,
                                                             A,
                                                             Atype,
                                                             lda,
                                                             B,
                                                             Btype,
                                                             ldb,
                                                             beta,
                                                             C,
                                                             Ctype,
                                                             ldc));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasSgemmEx is not supported on musa mp31"));
#endif
  }

  static void GEMM_EX_64(phi::GPUContext *dev_ctx,
                         mublasOperation_t transa,
                         mublasOperation_t transb,
                         int64_t m,
                         int64_t n,
                         int64_t k,
                         const float *alpha,
                         const void *A,
                         musaDataType_t Atype,
                         int64_t lda,
                         const void *B,
                         musaDataType_t Btype,
                         int64_t ldb,
                         const float *beta,
                         void *C,
                         musaDataType_t Ctype,
                         int64_t ldc) {
// Because the gcc 4.8 doesn't expand template parameter pack that
// appears in a lambda-expression, I can not use template parameter pack
// here.
#if !defined(PADDLE_WITH_MUSA) && defined(__linux__)
    VLOG(5) << "use_tensor_op_math: "
            << (dev_ctx->tensor_core_available() ? "True" : "False");
    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgemmEx_64(handle,
                                                                transa,
                                                                transb,
                                                                m,
                                                                n,
                                                                k,
                                                                alpha,
                                                                A,
                                                                Atype,
                                                                lda,
                                                                B,
                                                                Btype,
                                                                ldb,
                                                                beta,
                                                                C,
                                                                Ctype,
                                                                ldc));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasSgemmEx_64 is not supported on musa mp31"));
#endif
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasStrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgetrfBatched(args...));
    PADDLE_THROW(
          common::errors::Unimplemented("currently there are not mublasSgetrfBatched."));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgetriBatched(args...));
    PADDLE_THROW(
          common::errors::Unimplemented("currently there are not mublasSgetriBatched."));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSmatinvBatched(args...));
    PADDLE_THROW(
          common::errors::Unimplemented("currently there are not mublasSmatinvBatched."));
  }

  template <typename... ARGS>
  static void GETRS_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasSgetrsBatched(args...));
    PADDLE_THROW(
          common::errors::Unimplemented("currently there are not mublasSgetrsBatched."));
  }

  template <typename... ARGS>
  static void TRSM_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasStrsmBatched(args...));
  }
};

template <>
struct CUBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDaxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDcopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgemmBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "DgemmBatched is not supported on musa mp31"));
#endif
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::mublasDgemmStridedBatched(args...));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "DgemmStridedBatched is not supported on musa mp31"));
#endif
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Currently there are not mublasDgemmEx."));
  }

  template <typename... ARGS>
  static void GEMM_EX_64(ARGS... args UNUSED) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Currently there are not mublasDgemmEx_64."));
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDtrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgetrfBatched(args...));
    PADDLE_THROW(
          common::errors::Unimplemented("currently there are not mublasDgetrfBatched."));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgetriBatched(args...));
    PADDLE_THROW(
          common::errors::Unimplemented("currently there are not mublasDgetrfBatched."));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDmatinvBatched(args...));
    PADDLE_THROW(
          common::errors::Unimplemented("currently there are not mublasDmatinvBatched."));
  }

  template <typename... ARGS>
  static void GETRS_BATCH(ARGS... args) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDgetrsBatched(args...));
    PADDLE_THROW(
          common::errors::Unimplemented("currently there are not mublasDgetrsBatched."));
  }

  template <typename... ARGS>
  static void TRSM_BATCH(ARGS... args) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasDtrsmBatched(args...));
  }
};

template <>
struct CUBlas<phi::float16> {
  using float16 = phi::float16;

  static void GEMM(mublasHandle_t handle,
                   mublasOperation_t transa,
                   mublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const float16 *alpha,
                   const float16 *A,
                   int lda,
                   const float16 *B,
                   int ldb,
                   const float16 *beta,
                   float16 *C,
                   int ldc) {
    // PADDLE_ENFORCE_GPU_SUCCESS(
    //     phi::dynload::mublasHgemm(handle,
    //                               transa,
    //                               transb,
    //                               m,
    //                               n,
    //                               k,
    //                               reinterpret_cast<const __half *>(alpha),
    //                               reinterpret_cast<const __half *>(A),
    //                               lda,
    //                               reinterpret_cast<const __half *>(B),
    //                               ldb,
    //                               reinterpret_cast<const __half *>(beta),
    //                               reinterpret_cast<__half *>(C),
    //                               ldc));
    PADDLE_THROW(
          common::errors::Unimplemented("currently there are no mublasHgemm."));
  }

#if defined(__MUSACC__)
  static void GEMM_BATCH(phi::GPUContext *dev_ctx,
                         mublasOperation_t transa,
                         mublasOperation_t transb,
                         int m,
                         int n,
                         int k,
                         const float16 *alpha,
                         const float16 **A,
                         musaDataType_t Atype,
                         int lda,
                         const float16 **B,
                         musaDataType_t Btype,
                         int ldb,
                         const float16 *beta,
                         float16 **C,
                         musaDataType_t Ctype,
                         int ldc,
                         int batchCount,
                         mublasComputeType_t computeType) {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
    thrust::device_vector<const void *> A_ptr(A, A + batchCount);
    thrust::device_vector<const void *> B_ptr(B, B + batchCount);
    thrust::device_vector<void *> C_ptr(C, C + batchCount);
    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      bool use_tensor_op_math = dev_ctx->tensor_core_available();
      if (use_tensor_op_math) {
        algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
      }
      VLOG(5) << "use_tensor_op_math: "
              << (use_tensor_op_math ? "True" : "False");
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmBatchedEx(handle,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A_ptr.data().get(),
                                            Atype,
                                            lda,
                                            B_ptr.data().get(),
                                            Btype,
                                            ldb,
                                            beta,
                                            C_ptr.data().get(),
                                            Ctype,
                                            ldc,
                                            batchCount,
                                            computeType,
                                            algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmBatchedEx is not supported on musa mp31"));
#endif
  }
#endif

  static void GEMM_STRIDED_BATCH(mublasHandle_t handle,
                                 mublasOperation_t transa,
                                 mublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const float16 *alpha,
                                 const float16 *A,
                                 int lda,
                                 long long int strideA,  // NOLINT
                                 const float16 *B,       // NOLINT
                                 int ldb,
                                 long long int strideB,  // NOLINT
                                 const float16 *beta,
                                 float16 *C,
                                 int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
#if !defined(PADDLE_WITH_MUSA)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasHgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const __half *>(alpha),
        reinterpret_cast<const __half *>(A),
        lda,
        strideA,
        reinterpret_cast<const __half *>(B),
        ldb,
        strideB,
        reinterpret_cast<const __half *>(beta),
        reinterpret_cast<__half *>(C),
        ldc,
        strideC,
        batchCount));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "HgemmStridedBatched is not supported on musa mp31"));
#endif
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/mublas/index.html#mublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      mublasOperation_t transa,
                      mublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      musaDataType_t Atype,
                      int lda,
                      const void *B,
                      musaDataType_t Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      musaDataType_t Ctype,
                      int ldc,
                      mublasComputeType_t computeType) {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;

    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      bool use_tensor_op_math = dev_ctx->tensor_core_available();
      if (use_tensor_op_math) {
        algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
      }
      VLOG(5) << "Float16 GemmEx params: transa=" << transa << " transb=" << transb
              << " m=" << m << " n=" << n << " k=" << k
              << " lda=" << lda << " ldb=" << ldb << " ldc=" << ldc
              << " A=" << A << " B=" << B << " C=" << C
              << " Atype=" << Atype << " Btype=" << Btype
              << " Ctype=" << Ctype << " computeType=" << computeType
              << " algo=" << algo
              << " alpha=" << *(float*)alpha << " beta=" << *(float*)beta;
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasGemmEx(handle,
                                                            transa,
                                                            transb,
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            Atype,
                                                            lda,
                                                            B,
                                                            Btype,
                                                            ldb,
                                                            beta,
                                                            C,
                                                            Ctype,
                                                            ldc,
                                                            computeType,
                                                            algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmEx is not supported on musa mp31"));
#endif
  }

  static void GEMM_EX_64(phi::GPUContext *dev_ctx,
                         mublasOperation_t transa,
                         mublasOperation_t transb,
                         int64_t m,
                         int64_t n,
                         int64_t k,
                         const void *alpha,
                         const void *A,
                         musaDataType_t Atype,
                         int64_t lda,
                         const void *B,
                         musaDataType_t Btype,
                         int64_t ldb,
                         const void *beta,
                         void *C,
                         musaDataType_t Ctype,
                         int64_t ldc,
                         musaDataType_t computeType) {
#if CUDA_VERSION >= 12030 && defined(__linux__) && !defined(PADDLE_WITH_MUSA)
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    mublasComputeType_t migratedComputeType = MUBLAS_COMPUTE_32F;
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmEx_64(handle,
                                        transa,
                                        transb,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        Atype,
                                        lda,
                                        B,
                                        Btype,
                                        ldb,
                                        beta,
                                        C,
                                        Ctype,
                                        ldc,
                                        migratedComputeType,
                                        algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmEx_64 is not supported on musa mp31"));
#endif
  }
};

template <>
struct CUBlas<phi::complex64> {
  static void GEMV(mublasHandle_t handle,
                   mublasOperation_t transa,
                   int m,
                   int n,
                   const phi::complex64 *alpha,
                   const phi::complex64 *A,
                   int lda,
                   const phi::complex64 *B,
                   int ldb,
                   const phi::complex64 *beta,
                   phi::complex64 *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCgemv(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex *>(A),
        lda,
        reinterpret_cast<const muFloatComplex *>(B),
        ldb,
        reinterpret_cast<const muFloatComplex *>(beta),
        reinterpret_cast<muFloatComplex *>(C),
        ldc));
  }

  static void AXPY(mublasHandle_t handle,
                   int n,
                   const phi::complex64 *alpha,
                   const phi::complex64 *X,
                   const int incX,
                   phi::complex64 *Y,
                   const int incY) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCaxpy(
        handle,
        n,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex *>(X),
        incX,
        reinterpret_cast<muFloatComplex *>(Y),
        incY));
  }

  static void GEMM_STRIDED_BATCH(mublasHandle_t handle,
                                 mublasOperation_t transa,
                                 mublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const phi::complex64 *alpha,
                                 const phi::complex64 *A,
                                 int lda,
                                 long long int strideA,    // NOLINT
                                 const phi::complex64 *B,  // NOLINT
                                 int ldb,
                                 long long int strideB,  // NOLINT
                                 const phi::complex64 *beta,
                                 phi::complex64 *C,
                                 int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex *>(A),
        lda,
        strideA,
        reinterpret_cast<const muFloatComplex *>(B),
        ldb,
        strideB,
        reinterpret_cast<const muFloatComplex *>(beta),
        reinterpret_cast<muFloatComplex *>(C),
        ldc,
        strideC,
        batchCount));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "CgemmStridedBatched is not supported on musa mp31"));
#endif
  }

  static void GEMM(mublasHandle_t handle,
                   mublasOperation_t transa,
                   mublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const phi::complex64 *alpha,
                   const phi::complex64 *A,
                   int lda,
                   const phi::complex64 *B,
                   int ldb,
                   const phi::complex64 *beta,
                   phi::complex64 *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex *>(A),
        lda,
        reinterpret_cast<const muFloatComplex *>(B),
        ldb,
        reinterpret_cast<const muFloatComplex *>(beta),
        reinterpret_cast<muFloatComplex *>(C),
        ldc));
  }

  static void TRSM(mublasHandle_t handle,
                   mublasSideMode_t side,
                   mublasFillMode_t uplo,
                   mublasOperation_t transa,
                   mublasDiagType_t diag,
                   int m,
                   int n,
                   const phi::complex64 *alpha,
                   const phi::complex64 *A,
                   int lda,
                   phi::complex64 *B,
                   int ldb) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCtrsm(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex *>(A),
        lda,
        reinterpret_cast<muFloatComplex *>(B),
        ldb));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/mublas/index.html#mublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      mublasOperation_t transa,
                      mublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      musaDataType_t Atype,
                      int lda,
                      const void *B,
                      musaDataType_t Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      musaDataType_t Ctype,
                      int ldc,
                      musaDataType_t computeType) {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
#if CUDA_VERSION >= 9000 || defined(PADDLE_WITH_MUSA)
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
#endif  // CUDA_VERSION >= 9000

    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasGemmEx(handle,
                                                            transa,
                                                            transb,
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            Atype,
                                                            lda,
                                                            B,
                                                            Btype,
                                                            ldb,
                                                            beta,
                                                            C,
                                                            Ctype,
                                                            ldc,
                                                            computeType,
                                                            algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmEx is not supported on musa mp31"));
#endif
  }

  static void GEMM_EX_64(phi::GPUContext *dev_ctx,
                         mublasOperation_t transa,
                         mublasOperation_t transb,
                         int64_t m,
                         int64_t n,
                         int64_t k,
                         const void *alpha,
                         const void *A,
                         musaDataType_t Atype,
                         int64_t lda,
                         const void *B,
                         musaDataType_t Btype,
                         int64_t ldb,
                         const void *beta,
                         void *C,
                         musaDataType_t Ctype,
                         int64_t ldc,
                         musaDataType_t computeType) {
#if CUDA_VERSION >= 12030 && defined(__linux__) && !defined(PADDLE_WITH_MUSA)
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
    mublasComputeType_t migratedComputeType = MUBLAS_COMPUTE_32F;
    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmEx_64(handle,
                                        transa,
                                        transb,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        Atype,
                                        lda,
                                        B,
                                        Btype,
                                        ldb,
                                        beta,
                                        C,
                                        Ctype,
                                        ldc,
                                        migratedComputeType,
                                        algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmEx_64 is not supported on musa mp31"));
#endif
  }

  static void TRSM_BATCH(mublasHandle_t handle,
                         mublasSideMode_t side,
                         mublasFillMode_t uplo,
                         mublasOperation_t transa,
                         mublasDiagType_t diag,
                         int m,
                         int n,
                         const phi::complex64 *alpha,
                         const phi::complex64 **A,
                         int lda,
                         phi::complex64 **B,
                         int ldb,
                         int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCtrsmBatched(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const muFloatComplex *>(alpha),
        reinterpret_cast<const muFloatComplex **>(A),
        lda,
        reinterpret_cast<muFloatComplex **>(B),
        ldb,
        batch_size));
  }

  static void GETRF_BATCH(mublasHandle_t handle,
                          int n,
                          phi::complex64 **A,
                          int lda,
                          int *ipiv,
                          int *info,
                          int batch_size) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCgetrfBatched(
    //     handle,
    //     n,
    //     reinterpret_cast<muFloatComplex **>(A),
    //     lda,
    //     ipiv,
    //     info,
    //     batch_size));
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasCgetrfBatched is not supported on musa mp_31"));
  }

  static void GETRI_BATCH(mublasHandle_t handle,
                          int n,
                          const phi::complex64 **A,
                          int lda,
                          const int *ipiv,
                          phi::complex64 **Ainv,
                          int ldc,
                          int *info,
                          int batch_size) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCgetriBatched(
    //     handle,
    //     n,
    //     reinterpret_cast<const muFloatComplex **>(A),
    //     lda,
    //     ipiv,
    //     reinterpret_cast<muFloatComplex **>(Ainv),
    //     ldc,
    //     info,
    //     batch_size));
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasCgetriBatched is not supported on musa mp_31"));
  }

  static void MATINV_BATCH(mublasHandle_t handle,
                           int n,
                           const phi::complex64 **A,
                           int lda,
                           phi::complex64 **Ainv,
                           int lda_inv,
                           int *info,
                           int batch_size) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasCmatinvBatched(
    //     handle,
    //     n,
    //     reinterpret_cast<const muFloatComplex **>(A),
    //     lda,
    //     reinterpret_cast<muFloatComplex **>(Ainv),
    //     lda_inv,
    //     info,
    //     batch_size));
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasCmatinvBatched is not supported on musa mp_31"));
  }
};

template <>
struct CUBlas<phi::complex128> {
  static void GEMV(mublasHandle_t handle,
                   mublasOperation_t transa,
                   int m,
                   int n,
                   const phi::complex128 *alpha,
                   const phi::complex128 *A,
                   int lda,
                   const phi::complex128 *B,
                   int ldb,
                   const phi::complex128 *beta,
                   phi::complex128 *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZgemv(
        handle,
        transa,
        m,
        n,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex *>(A),
        lda,
        reinterpret_cast<const muDoubleComplex *>(B),
        ldb,
        reinterpret_cast<const muDoubleComplex *>(beta),
        reinterpret_cast<muDoubleComplex *>(C),
        ldc));
  }

  static void AXPY(mublasHandle_t handle,
                   int n,
                   const phi::complex128 *alpha,
                   const phi::complex128 *X,
                   const int incX,
                   phi::complex128 *Y,
                   const int incY) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZaxpy(
        handle,
        n,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex *>(X),
        incX,
        reinterpret_cast<muDoubleComplex *>(Y),
        incY));
  }

  static void GEMM_STRIDED_BATCH(mublasHandle_t handle,
                                 mublasOperation_t transa,
                                 mublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const phi::complex128 *alpha,
                                 const phi::complex128 *A,
                                 int lda,
                                 long long int strideA,     // NOLINT
                                 const phi::complex128 *B,  // NOLINT
                                 int ldb,
                                 long long int strideB,  // NOLINT
                                 const phi::complex128 *beta,
                                 phi::complex128 *C,
                                 int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZgemmStridedBatched(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex *>(A),
        lda,
        strideA,
        reinterpret_cast<const muDoubleComplex *>(B),
        ldb,
        strideB,
        reinterpret_cast<const muDoubleComplex *>(beta),
        reinterpret_cast<muDoubleComplex *>(C),
        ldc,
        strideC,
        batchCount));
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "CgemmStridedBatched is not supported on musa mp31"));
#endif
  }

  static void GEMM(mublasHandle_t handle,
                   mublasOperation_t transa,
                   mublasOperation_t transb,
                   int m,
                   int n,
                   int k,
                   const phi::complex128 *alpha,
                   const phi::complex128 *A,
                   int lda,
                   const phi::complex128 *B,
                   int ldb,
                   const phi::complex128 *beta,
                   phi::complex128 *C,
                   int ldc) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZgemm(
        handle,
        transa,
        transb,
        m,
        n,
        k,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex *>(A),
        lda,
        reinterpret_cast<const muDoubleComplex *>(B),
        ldb,
        reinterpret_cast<const muDoubleComplex *>(beta),
        reinterpret_cast<muDoubleComplex *>(C),
        ldc));
  }

  static void TRSM(mublasHandle_t handle,
                   mublasSideMode_t side,
                   mublasFillMode_t uplo,
                   mublasOperation_t transa,
                   mublasDiagType_t diag,
                   int m,
                   int n,
                   const phi::complex128 *alpha,
                   const phi::complex128 *A,
                   int lda,
                   phi::complex128 *B,
                   int ldb) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZtrsm(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex *>(A),
        lda,
        reinterpret_cast<muDoubleComplex *>(B),
        ldb));
  }

  static void TRSM_BATCH(mublasHandle_t handle,
                         mublasSideMode_t side,
                         mublasFillMode_t uplo,
                         mublasOperation_t transa,
                         mublasDiagType_t diag,
                         int m,
                         int n,
                         const phi::complex128 *alpha,
                         const phi::complex128 **A,
                         int lda,
                         phi::complex128 **B,
                         int ldb,
                         int batch_size) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZtrsmBatched(
        handle,
        side,
        uplo,
        transa,
        diag,
        m,
        n,
        reinterpret_cast<const muDoubleComplex *>(alpha),
        reinterpret_cast<const muDoubleComplex **>(A),
        lda,
        reinterpret_cast<muDoubleComplex **>(B),
        ldb,
        batch_size));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/mublas/index.html#mublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(phi::GPUContext *dev_ctx,
                      mublasOperation_t transa,
                      mublasOperation_t transb,
                      int m,
                      int n,
                      int k,
                      const void *alpha,
                      const void *A,
                      musaDataType_t Atype,
                      int lda,
                      const void *B,
                      musaDataType_t Btype,
                      int ldb,
                      const void *beta,
                      void *C,
                      musaDataType_t Ctype,
                      int ldc,
                      mublasComputeType_t computeType) {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      bool use_tensor_op_math = dev_ctx->tensor_core_available();
      if (use_tensor_op_math) {
        algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
      }
      VLOG(5) << "use_tensor_op_math: "
              << (use_tensor_op_math ? "True" : "False");
      PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasGemmEx(handle,
                                                            transa,
                                                            transb,
                                                            m,
                                                            n,
                                                            k,
                                                            alpha,
                                                            A,
                                                            Atype,
                                                            lda,
                                                            B,
                                                            Btype,
                                                            ldb,
                                                            beta,
                                                            C,
                                                            Ctype,
                                                            ldc,
                                                            computeType,
                                                            algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmEx is not supported on musa mp31"));
#endif
  }

  static void GEMM_EX_64(phi::GPUContext *dev_ctx,
                         mublasOperation_t transa,
                         mublasOperation_t transb,
                         int64_t m,
                         int64_t n,
                         int64_t k,
                         const void *alpha,
                         const void *A,
                         musaDataType_t Atype,
                         int64_t lda,
                         const void *B,
                         musaDataType_t Btype,
                         int64_t ldb,
                         const void *beta,
                         void *C,
                         musaDataType_t Ctype,
                         int64_t ldc,
                         musaDataType_t computeType) {
#if CUDA_VERSION >= 12030 && defined(__linux__) && !defined(PADDLE_WITH_MUSA)
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
    mublasComputeType_t migratedComputeType = MUBLAS_COMPUTE_32F;
    dev_ctx->TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmEx_64(handle,
                                        transa,
                                        transb,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        A,
                                        Atype,
                                        lda,
                                        B,
                                        Btype,
                                        ldb,
                                        beta,
                                        C,
                                        Ctype,
                                        ldc,
                                        migratedComputeType,
                                        algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmEx_64 is not supported on musa mp 31"));
#endif
  }

  static void GETRF_BATCH(mublasHandle_t handle,
                          int n,
                          phi::complex128 **A,
                          int lda,
                          int *ipiv,
                          int *info,
                          int batch_size) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZgetrfBatched(
    //     handle,
    //     n,
    //     reinterpret_cast<muDoubleComplex **>(A),
    //     lda,
    //     ipiv,
    //     info,
    //     batch_size));
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmEx_64 is not supported on musa mp 31"));
  }

  static void GETRI_BATCH(mublasHandle_t handle,
                          int n,
                          const phi::complex128 **A,
                          int lda,
                          const int *ipiv,
                          phi::complex128 **Ainv,
                          int ldc,
                          int *info,
                          int batch_size) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZgetriBatched(
    //     handle,
    //     n,
    //     reinterpret_cast<const muDoubleComplex **>(A),
    //     lda,
    //     ipiv,
    //     reinterpret_cast<muDoubleComplex **>(Ainv),
    //     ldc,
    //     info,
    //     batch_size));
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasZgetriBatched is not supported on musa mp 31"));
  }

  static void MATINV_BATCH(mublasHandle_t handle,
                           int n,
                           const phi::complex128 **A,
                           int lda,
                           phi::complex128 **Ainv,
                           int lda_inv,
                           int *info,
                           int batch_size) {
    // PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasZmatinvBatched(
    //     handle,
    //     n,
    //     reinterpret_cast<const muDoubleComplex **>(A),
    //     lda,
    //     reinterpret_cast<muDoubleComplex **>(Ainv),
    //     lda_inv,
    //     info,
    //     batch_size));
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmEx_64 is not supported on musa mp 31"));
  }
};

inline void CheckGEMMNSize(int64_t N) {
  constexpr int64_t kMaxN = 1073741823;
  if (N > kMaxN) {
    PADDLE_THROW(common::errors::Unimplemented(
        "mublas GEMM does not support N > %ld. Got N = %ld. ", kMaxN, N));
  }
}

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
                                 int64_t M,
                                 int64_t N,
                                 int64_t K,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
// #if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
  if (FLAGS_enable_cublas_tensor_op_math && std::is_same<T, float>::value) {
    auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
    if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__) && !defined(PADDLE_WITH_MUSA) 
      CUBlas<T>::GEMM_EX_64(&cuda_ctx,
                            cuTransB,
                            cuTransA,
                            N,
                            M,
                            K,
                            &alpha,
                            B,
                            MUSA_R_32F,
                            ldb,
                            A,
                            MUSA_R_32F,
                            lda,
                            &beta,
                            C,
                            MUSA_R_32F,
                            N);
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "GEMM_EX_64 is not supported on musa mp31"));
#endif
    } else {
      CheckGEMMNSize(N);
      CUBlas<T>::GEMM_EX(&cuda_ctx,
                         cuTransB,
                         cuTransA,
                         static_cast<int>(N),
                         static_cast<int>(M),
                         static_cast<int>(K),
                         &alpha,
                         B,
                         MUSA_R_32F,
                         static_cast<int>(ldb),
                         A,
                         MUSA_R_32F,
                         static_cast<int>(lda),
                         &beta,
                         C,
                         MUSA_R_32F,
                         static_cast<int>(N));
    }
  } else {
// #endif  // CUDA_VERSION >= 8000
    if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
      PADDLE_THROW(common::errors::Unimplemented(
          "GEMM_EX_64 is not supported on musa mp31"));
    } else {
      dev_ctx_.CublasCall([&](mublasHandle_t handle) {
        CUBlas<T>::GEMM(handle,
                        cuTransB,
                        cuTransA,
                        static_cast<int>(N),
                        static_cast<int>(M),
                        static_cast<int>(K),
                        &alpha,
                        B,
                        static_cast<int>(ldb),
                        A,
                        static_cast<int>(lda),
                        &beta,
                        C,
                        static_cast<int>(N));
      });
    }

// #if CUDA_VERSION >= 8000
  }
// #endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        phi::float16 alpha,
                                        const phi::float16 *A,
                                        const phi::float16 *B,
                                        phi::float16 beta,
                                        phi::float16 *C) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      31,
      common::errors::InvalidArgument(
          "mublas fp16 gemm requires musa compute capability >= 31,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
  // mublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use mublasGemmEx instead which does pseudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__) && !defined(PADDLE_WITH_MUSA)
    CUBlas<phi::float16>::GEMM_EX_64(&cuda_ctx,
                                     cuTransB,
                                     cuTransA,
                                     N,
                                     M,
                                     K,
                                     &h_alpha,
                                     B,
                                     MUSA_R_16F,
                                     ldb,
                                     A,
                                     MUSA_R_16F,
                                     lda,
                                     &h_beta,
                                     C,
                                     MUSA_R_16F,
                                     N,
                                     MUSA_R_32F);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "GEMM_EX_64 is not supported on musa mp31"));
#endif  // CUDA_VERSION >= 12030
  } else {
    CheckGEMMNSize(N);
    CUBlas<phi::float16>::GEMM_EX(&cuda_ctx,
                                  cuTransB,
                                  cuTransA,
                                  static_cast<int>(N),
                                  static_cast<int>(M),
                                  static_cast<int>(K),
                                  &alpha,
                                  B,
                                  MUSA_R_16F,
                                  static_cast<int>(ldb),
                                  A,
                                  MUSA_R_16F,
                                  static_cast<int>(lda),
                                  &beta,
                                  C,
                                  MUSA_R_16F,
                                  static_cast<int>(N),
                                  MUBLAS_COMPUTE_16F);
  }
#else
  // CUDA 7.5 does not support mublasGemmEx, hence we fall back to use hgemm
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
    PADDLE_THROW(common::errors::Unimplemented(
        "GEMM_EX_64 is not supported on musa mp31"));
  } else {
    dev_ctx_.CublasCall([&](mublasHandle_t handle) {
      CUBlas<phi::float16>::GEMM(handle,
                                 cuTransB,
                                 cuTransA,
                                 static_cast<int>(N),
                                 static_cast<int>(M),
                                 static_cast<int>(K),
                                 &h_alpha,
                                 h_B,
                                 static_cast<int>(ldb),
                                 h_A,
                                 static_cast<int>(lda),
                                 &h_beta,
                                 h_C,
                                 static_cast<int>(N));
    });
  }
#endif  // CUDA_VERSION >= 8000
}

template <>
template <typename T, typename U>
void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                 CBLAS_TRANSPOSE transB,
                                 int64_t M,
                                 int64_t N,
                                 int64_t K,
                                 U alpha,
                                 const T *A,
                                 const T *B,
                                 U beta,
                                 T *C) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;

  T t_alpha = static_cast<T>(alpha);
  T t_beta = static_cast<T>(beta);

#if CUDA_VERSION >= 8000
  if (FLAGS_enable_cublas_tensor_op_math && std::is_same<T, float>::value) {
    auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
    if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__) && !defined(PADDLE_WITH_MUSA)
      CUBlas<T>::GEMM_EX_64(&cuda_ctx,
                            cuTransB,
                            cuTransA,
                            N,
                            M,
                            K,
                            &t_alpha,
                            B,
                            MUSA_R_32F,
                            ldb,
                            A,
                            MUSA_R_32F,
                            lda,
                            &t_beta,
                            C,
                            MUSA_R_32F,
                            N);
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "GEMM_EX_64 is not supported on musa mp31"));
#endif
    } else {
      CheckGEMMNSize(N);
      CUBlas<T>::GEMM_EX(&cuda_ctx,
                         cuTransB,
                         cuTransA,
                         static_cast<int>(N),
                         static_cast<int>(M),
                         static_cast<int>(K),
                         &t_alpha,
                         B,
                         MUSA_R_32F,
                         static_cast<int>(ldb),
                         A,
                         MUSA_R_32F,
                         static_cast<int>(lda),
                         &t_beta,
                         C,
                         MUSA_R_32F,
                         static_cast<int>(N));
    }
  } else {
#endif  // CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
      PADDLE_THROW(common::errors::Unimplemented(
          "GEMM_EX_64 is not supported on musa mp31"));
    } else {
      dev_ctx_.CublasCall([&](mublasHandle_t handle) {
        CUBlas<T>::GEMM(handle,
                        cuTransB,
                        cuTransA,
                        static_cast<int>(N),
                        static_cast<int>(M),
                        static_cast<int>(K),
                        &t_alpha,
                        B,
                        static_cast<int>(ldb),
                        A,
                        static_cast<int>(lda),
                        &t_beta,
                        C,
                        static_cast<int>(N));
      });
    }

#if CUDA_VERSION >= 8000
  }
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        float alpha,
                                        const phi::float16 *A,
                                        const phi::float16 *B,
                                        float beta,
                                        phi::float16 *C) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      31,
      common::errors::InvalidArgument(
          "mublas fp16 gemm requires GPU compute capability >= 31,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  float h_alpha = alpha;
  float h_beta = beta;

#if CUDA_VERSION >= 8000 | defined(PADDLE_WITH_MUSA)
  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
#endif
  // mublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use mublasGemmEx instead which does pseudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__) && !defined(PADDLE_WITH_MUSA)
    CUBlas<phi::float16>::GEMM_EX_64(&cuda_ctx,
                                     cuTransB,
                                     cuTransA,
                                     N,
                                     M,
                                     K,
                                     &h_alpha,
                                     B,
                                     MUSA_R_16F,
                                     ldb,
                                     A,
                                     MUSA_R_16F,
                                     lda,
                                     &h_beta,
                                     C,
                                     MUSA_R_16F,
                                     N,
                                     MUSA_R_32F);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "GEMM_EX_64 is not supported on musa mp31"));
#endif  // CUDA_VERSION >= 12030
  } else {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    CheckGEMMNSize(N);
    CUBlas<phi::float16>::GEMM_EX(&cuda_ctx,
                                  cuTransB,
                                  cuTransA,
                                  static_cast<int>(N),
                                  static_cast<int>(M),
                                  static_cast<int>(K),
                                  &h_alpha,
                                  B,
                                  MUSA_R_16F,
                                  static_cast<int>(ldb),
                                  A,
                                  MUSA_R_16F,
                                  static_cast<int>(lda),
                                  &h_beta,
                                  C,
                                  MUSA_R_16F,
                                  static_cast<int>(N),
                                  MUBLAS_COMPUTE_32F);
#else
    // CUDA 7.5 does not support mublasGemmEx, hence we fall back to use hgemm
    dev_ctx_.CublasCall([&](mublasHandle_t handle) {
      CUBlas<phi::float16>::GEMM(handle,
                                 cuTransB,
                                 cuTransA,
                                 static_cast<int>(N),
                                 static_cast<int>(M),
                                 static_cast<int>(K),
                                 &h_alpha,
                                 h_B,
                                 static_cast<int>(ldb),
                                 h_A,
                                 static_cast<int>(lda),
                                 &h_beta,
                                 h_C,
                                 static_cast<int>(N));
    });
#endif  // CUDA_VERSION >= 8000
  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        phi::bfloat16 alpha,
                                        const phi::bfloat16 *A,
                                        const phi::bfloat16 *B,
                                        phi::bfloat16 beta,
                                        phi::bfloat16 *C) const {
#if CUDA_VERSION >= 11000 || defined(PADDLE_WITH_MUSA)
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      31,
      common::errors::InvalidArgument(
          "mublas bf16 gemm requires GPU compute capability >= 31,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
  
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__) && defined(PADDLE_WITH_MUSA)
    mublasComputeType_t migratedComputeType = MUBLAS_COMPUTE_32F;
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmEx_64(handle,
                                        cuTransB,
                                        cuTransA,
                                        N,
                                        M,
                                        K,
                                        &h_alpha,
                                        B,
                                        MUSA_R_16BF,
                                        ldb,
                                        A,
                                        MUSA_R_16BF,
                                        lda,
                                        &h_beta,
                                        C,
                                        MUSA_R_16BF,
                                        N,
                                        migratedComputeType,
                                        algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmEx_64 is not supported on musa mp31"));
#endif  // CUDA_VERSION >= 12030
  } else {
    CheckGEMMNSize(N);
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      bool use_tensor_op_math = dev_ctx_.tensor_core_available();
      if (use_tensor_op_math) {
        algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
      }
      VLOG(5) << "BFloat16 GemmEx params: transa=" << cuTransA << " transb=" << cuTransB
              << " m=" << M << " n=" << N << " k=" << K
              << " lda=" << lda << " ldb=" << ldb << " ldc=" << N
              << " A=" << A << " B=" << B << " C=" << C
              << " Atype=" << MUSA_R_16BF << " Btype=" << MUSA_R_16BF
              << " Ctype=" << MUSA_R_16BF << " computeType=" << MUBLAS_COMPUTE_32F
              << " algo=" << algo
              << " alpha=" << h_alpha << " beta=" << h_beta;
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmEx(handle,
                                     cuTransB,
                                     cuTransA,
                                     static_cast<int>(N),
                                     static_cast<int>(M),
                                     static_cast<int>(K),
                                     &h_alpha,
                                     B,
                                     MUSA_R_16BF,
                                     static_cast<int>(ldb),
                                     A,
                                     MUSA_R_16BF,
                                     static_cast<int>(lda),
                                     &h_beta,
                                     C,
                                     MUSA_R_16BF,
                                     static_cast<int>(N),
                                     MUBLAS_COMPUTE_32F,
                                     algo));
    });
  }
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "mublasGemmEx with bfloat16 is not supported on musa mp31"));

#endif  // CUDA_VERSION >= 11000
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        float alpha,
                                        const phi::bfloat16 *A,
                                        const phi::bfloat16 *B,
                                        float beta,
                                        phi::bfloat16 *C) const {
#if CUDA_VERSION >= 11000 || defined(PADDLE_WITH_MUSA)
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      31,
      common::errors::InvalidArgument(
          "mublas bf16 gemm requires GPU compute capability >= 80,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  float h_alpha = alpha;
  float h_beta = beta;

  mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
  
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__) && !defined(PADDLE_WITH_MUSA)
    mublasComputeType_t migratedComputeType = MUBLAS_COMPUTE_32F;
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmEx_64(handle,
                                        cuTransB,
                                        cuTransA,
                                        N,
                                        M,
                                        K,
                                        &h_alpha,
                                        B,
                                        MUSA_R_16BF,
                                        ldb,
                                        A,
                                        MUSA_R_16BF,
                                        lda,
                                        &h_beta,
                                        C,
                                        MUSA_R_16BF,
                                        N,
                                        migratedComputeType,
                                        algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmEx_64 is not supported on musa mp31"));
#endif  // CUDA_VERSION >= 12030
  } else {
    CheckGEMMNSize(N);
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      bool use_tensor_op_math = dev_ctx_.tensor_core_available();
      if (use_tensor_op_math) {
        algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
      }
      
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmEx(handle,
                                     cuTransB,
                                     cuTransA,
                                     static_cast<int>(N),
                                     static_cast<int>(M),
                                     static_cast<int>(K),
                                     &h_alpha,
                                     B,
                                     MUSA_R_16BF,
                                     static_cast<int>(ldb),
                                     A,
                                     MUSA_R_16BF,
                                     static_cast<int>(lda),
                                     &h_beta,
                                     C,
                                     MUSA_R_16BF,
                                     static_cast<int>(N),
                                     MUBLAS_COMPUTE_32F,
                                     algo));
    });
  }
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "mublasGemmEx with bfloat16 is not supported on musa mp31"));

#endif  // CUDA_VERSION >= 11000
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        phi::complex64 alpha,
                                        const phi::complex64 *A,
                                        const phi::complex64 *B,
                                        phi::complex64 beta,
                                        phi::complex64 *C) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      31,
      common::errors::InvalidArgument(
          "mublas complex64 gemm requires GPU compute capability >= 31,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  thrust::complex<float> c_alpha =
      thrust::complex<float>(alpha.real, alpha.imag);
  thrust::complex<float> c_beta = thrust::complex<float>(beta.real, beta.imag);

#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
#endif

  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__) && !defined(PADDLE_WITH_MUSA)
    CUBlas<phi::complex64>::GEMM_EX_64(&cuda_ctx,
                                       cuTransB,
                                       cuTransA,
                                       N,
                                       M,
                                       K,
                                       &c_alpha,
                                       B,
                                       MUSA_C_32F,
                                       ldb,
                                       A,
                                       MUSA_C_32F,
                                       lda,
                                       &c_beta,
                                       C,
                                       MUSA_C_32F,
                                       N,
                                       MUSA_C_32F);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "GEMM_EX_64 is not supported on musa mp31"));
#endif  // CUDA_VERSION >= 12030
  } else {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    CheckGEMMNSize(N);
    CUBlas<phi::complex64>::GEMM_EX(&cuda_ctx,
                                    cuTransB,
                                    cuTransA,
                                    static_cast<int>(N),
                                    static_cast<int>(M),
                                    static_cast<int>(K),
                                    &c_alpha,
                                    B,
                                    MUSA_C_32F,
                                    static_cast<int>(ldb),
                                    A,
                                    MUSA_C_32F,
                                    static_cast<int>(lda),
                                    &c_beta,
                                    C,
                                    MUSA_C_32F,
                                    static_cast<int>(N),
                                    MUSA_C_32F);

#else
    dev_ctx_.CublasCall([&](mublasHandle_t handle) {
      CUBlas<phi::complex64>::GEMM(handle,
                                   cuTransB,
                                   cuTransA,
                                   static_cast<int>(N),
                                   static_cast<int>(M),
                                   static_cast<int>(K),
                                   &c_alpha,
                                   h_B,
                                   static_cast<int>(ldb),
                                   h_A,
                                   static_cast<int>(lda),
                                   &c_beta,
                                   h_C,
                                   static_cast<int>(N));
    });

#endif  // CUDA_VERSION >= 8000
  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        phi::complex128 alpha,
                                        const phi::complex128 *A,
                                        const phi::complex128 *B,
                                        phi::complex128 beta,
                                        phi::complex128 *C) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      31,
      common::errors::InvalidArgument(
          "mublas complex128 gemm requires GPU compute capability >= 31,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  thrust::complex<double> c_alpha =
      thrust::complex<double>(alpha.real, alpha.imag);
  thrust::complex<double> c_beta =
      thrust::complex<double>(beta.real, beta.imag);
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
#endif

  // mublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use mublasGemmEx instead which does pseudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if CUDA_VERSION >= 12030 && defined(__linux__) && !defined(PADDLE_WITH_MUSA)
    CUBlas<phi::complex128>::GEMM_EX_64(&cuda_ctx,
                                        cuTransB,
                                        cuTransA,
                                        N,
                                        M,
                                        K,
                                        &c_alpha,
                                        B,
                                        MUSA_C_64F,
                                        ldb,
                                        A,
                                        MUSA_C_64F,
                                        lda,
                                        &c_beta,
                                        C,
                                        MUSA_C_64F,
                                        N,
                                        MUSA_C_64F);
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "GEMM_EX_64 is not supported on musa mp31"));
#endif  // CUDA_VERSION >= 12030
  } else {
#if CUDA_VERSION >= 8000 || defined(PADDLE_WITH_MUSA)
    CheckGEMMNSize(N);
    CUBlas<phi::complex128>::GEMM_EX(&cuda_ctx,
                                     cuTransB,
                                     cuTransA,
                                     static_cast<int>(N),
                                     static_cast<int>(M),
                                     static_cast<int>(K),
                                     &c_alpha,
                                     B,
                                     MUSA_C_64F,
                                     static_cast<int>(ldb),
                                     A,
                                     MUSA_C_64F,
                                     static_cast<int>(lda),
                                     &c_beta,
                                     C,
                                     MUSA_C_64F,
                                     static_cast<int>(N),
                                     MUBLAS_COMPUTE_64F);
#else  // CUDA_VERSION >= 8000
    // CUDA 7.5 does not support mublasGemmEx, hence we fall back to use hgemm
    dev_ctx_.CublasCall([&](mublasHandle_t handle) {
      CUBlas<phi::complex128>::GEMM(handle,
                                    cuTransB,
                                    cuTransA,
                                    static_cast<int>(N),
                                    static_cast<int>(M),
                                    static_cast<int>(K),
                                    &c_alpha,
                                    h_B,
                                    static_cast<int>(ldb),
                                    h_A,
                                    static_cast<int>(lda),
                                    &c_beta,
                                    h_C,
                                    static_cast<int>(N));
    });
#endif
  }
}

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMM(bool transA,
                                 bool transB,
                                 int M,
                                 int N,
                                 int K,
                                 T alpha,
                                 const T *A,
                                 int lda,
                                 const T *B,
                                 int ldb,
                                 T beta,
                                 T *C,
                                 int ldc) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  mublasOperation_t cuTransA = transA ? MUBLAS_OP_T : MUBLAS_OP_N;
  mublasOperation_t cuTransB = transB ? MUBLAS_OP_T : MUBLAS_OP_N;

#if CUDA_VERSION >= 8000
  CheckGEMMNSize(N);
  if (FLAGS_enable_cublas_tensor_op_math && std::is_same<T, float>::value) {
    auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
    CUBlas<T>::GEMM_EX(&cuda_ctx,
                       cuTransB,
                       cuTransA,
                       N,
                       M,
                       K,
                       &alpha,
                       B,
                       MUSA_R_32F,
                       ldb,
                       A,
                       MUSA_R_32F,
                       lda,
                       &beta,
                       C,
                       MUSA_R_32F,
                       ldc);
  } else {
#endif  // CUDA_VERSION >= 8000

    dev_ctx_.CublasCall([&](mublasHandle_t handle) {
      CUBlas<T>::GEMM(handle,
                      cuTransB,
                      cuTransA,
                      N,
                      M,
                      K,
                      &alpha,
                      B,
                      ldb,
                      A,
                      lda,
                      &beta,
                      C,
                      ldc);
    });

#if CUDA_VERSION >= 8000
  }
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(bool transA,
                                        bool transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::float16 alpha,
                                        const phi::float16 *A,
                                        int lda,
                                        const phi::float16 *B,
                                        int ldb,
                                        phi::float16 beta,
                                        phi::float16 *C,
                                        int ldc) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  mublasOperation_t cuTransA = transA ? MUBLAS_OP_T : MUBLAS_OP_N;
  mublasOperation_t cuTransB = transB ? MUBLAS_OP_T : MUBLAS_OP_N;

  mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);
  bool use_tensor_op_math = dev_ctx_.tensor_core_available();
  VLOG(5) << "use_tensor_op_math is : " << use_tensor_op_math;
  if (use_tensor_op_math) {
    algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }
  CheckGEMMNSize(N);
  dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasGemmEx(handle,
                                                          cuTransB,
                                                          cuTransA,
                                                          N,
                                                          M,
                                                          K,
                                                          &h_alpha,
                                                          B,
                                                          MUSA_R_16F,
                                                          ldb,
                                                          A,
                                                          MUSA_R_16F,
                                                          lda,
                                                          &h_beta,
                                                          C,
                                                          MUSA_R_16F,
                                                          ldc,
                                                          MUSA_R_32F,
                                                          algo));
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMM(bool transA,
                                        bool transB,
                                        int M,
                                        int N,
                                        int K,
                                        phi::bfloat16 alpha,
                                        const phi::bfloat16 *A,
                                        int lda,
                                        const phi::bfloat16 *B,
                                        int ldb,
                                        phi::bfloat16 beta,
                                        phi::bfloat16 *C,
                                        int ldc) const {
#if CUDA_VERSION >= 11000 || defined(PADDLE_WITH_MUSA)
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  mublasOperation_t cuTransA = transA ? MUBLAS_OP_T : MUBLAS_OP_N;
  mublasOperation_t cuTransB = transB ? MUBLAS_OP_T : MUBLAS_OP_N;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      31,
      common::errors::InvalidArgument(
          "mublas bf16 gemm requires GPU compute capability >= 80,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
  bool use_tensor_op_math = dev_ctx_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }

  CheckGEMMNSize(N);
  dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasGemmEx(handle,
                                                          cuTransB,
                                                          cuTransA,
                                                          N,
                                                          M,
                                                          K,
                                                          &h_alpha,
                                                          B,
                                                          MUSA_R_16BF,
                                                          ldb,
                                                          A,
                                                          MUSA_R_16BF,
                                                          lda,
                                                          &h_beta,
                                                          C,
                                                          MUSA_R_16BF,
                                                          ldc,
                                                          MUSA_R_32F,
                                                          algo));
  });
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "mublasGemmEx with bfloat16 is not supported on musa mp31"));

#endif  // CUDA_VERSION >= 11000
}

template <>
template <typename T>
void Blas<phi::GPUContext>::AXPY(int n, T alpha, const T *x, T *y) const {
  dev_ctx_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::AXPY(handle, n, &alpha, x, 1, y, 1);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::SCAL(int n, const T alpha, T *x) const {
  dev_ctx_.CublasCall(
      [&](mublasHandle_t handle) { CUBlas<T>::SCAL(handle, n, &alpha, x, 1); });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::VCOPY(int n, const T *x, T *y) const {
  dev_ctx_.CublasCall(
      [&](mublasHandle_t handle) { CUBlas<T>::VCOPY(handle, n, x, 1, y, 1); });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                 int M,
                                 int N,
                                 T alpha,
                                 const T *A,
                                 const T *B,
                                 T beta,
                                 T *C) const {
  mublasOperation_t cuTransA = !trans_a ? MUBLAS_OP_T : MUBLAS_OP_N;

  dev_ctx_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::GEMV(handle, cuTransA, N, M, &alpha, A, N, B, 1, &beta, C, 1);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                        int M,
                                        int N,
                                        phi::float16 alpha,
                                        const phi::float16 *A,
                                        const phi::float16 *B,
                                        phi::float16 beta,
                                        phi::float16 *C) const {
  // Because mublas doesn't support half gemv, we use mublasHgemm to achieve it.
  if (trans_a) {
    this->template GEMM<phi::float16>(
        CblasNoTrans, CblasNoTrans, 1, N, M, alpha, B, A, beta, C);
  } else {
    this->template GEMM<phi::float16>(
        CblasNoTrans, CblasNoTrans, M, 1, N, alpha, A, B, beta, C);
  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::GEMV(bool trans_a,
                                        int M,
                                        int N,
                                        phi::bfloat16 alpha,
                                        const phi::bfloat16 *A,
                                        const phi::bfloat16 *B,
                                        phi::bfloat16 beta,
                                        phi::bfloat16 *C) const {
  // Because mublas doesn't support bfloat gemv, we use mublasHgemm to achieve
  // it.
  if (trans_a) {
    this->template GEMM<phi::bfloat16>(
        CblasNoTrans, CblasNoTrans, 1, N, M, alpha, B, A, beta, C);
  } else {
    this->template GEMM<phi::bfloat16>(
        CblasNoTrans, CblasNoTrans, M, 1, N, alpha, A, B, beta, C);
  }
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        T alpha,
                                        const T *A,
                                        const T *B,
                                        T beta,
                                        T *C,
                                        int64_t batchCount,
                                        int64_t strideA,
                                        int64_t strideB) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  int64_t ldc = N;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  const int64_t strideC = M * N;
  
  if ((FLAGS_enable_cublas_tensor_op_math && (std::is_same<T, float>::value)) ||
      std::is_same<T, phi::float16>::value) {
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;

    auto fp = std::is_same<T, float>::value ? MUSA_R_32F : MUSA_R_16F;
    auto compute_type = MUBLAS_COMPUTE_32F;

    float h_alpha = static_cast<float>(alpha);
    float h_beta = static_cast<float>(beta);
    void *a = static_cast<void *>(&h_alpha);
    void *b = static_cast<void *>(&h_beta);
    // set ComputeType as MUSA_R_32F for fp16, for better accuracy
    if (std::is_same<T, phi::float16>::value) {
      a = static_cast<void *>(&alpha);
      b = static_cast<void *>(&beta);
      compute_type = MUBLAS_COMPUTE_16F;
    }
    if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE) {
#if !defined(PADDLE_WITH_MUSA)
      dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::mublasGemmStridedBatchedEx_64(handle,
                                                        cuTransB,
                                                        cuTransA,
                                                        N,
                                                        M,
                                                        K,
                                                        a,
                                                        B,
                                                        fp,
                                                        ldb,
                                                        strideB,
                                                        A,
                                                        fp,
                                                        lda,
                                                        strideA,
                                                        b,
                                                        C,
                                                        fp,
                                                        ldc,
                                                        strideC,
                                                        batchCount,
                                                        compute_type,
                                                        algo));
      });
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "mublasGemmStridedBatchedEx_64 is not supported on musa mp31"));
#endif  // !defined(PADDLE_WITH_MUSA)
    } else {
      dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
        bool use_tensor_op_math = dev_ctx_.tensor_core_available();
        if (use_tensor_op_math) {
          algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
        }
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasGemmStridedBatchedEx(
            handle,
            cuTransB,
            cuTransA,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            a,
            B,
            fp,
            static_cast<int>(ldb),
            strideB,
            A,
            fp,
            static_cast<int>(lda),
            strideA,
            b,
            C,
            fp,
            static_cast<int>(ldc),
            strideC,
            static_cast<int>(batchCount),
            compute_type,
            algo));
      });
    }
  } else {
    dev_ctx_.CublasCall([&](mublasHandle_t handle) {
      CUBlas<T>::GEMM_STRIDED_BATCH(handle,
                                    cuTransB,
                                    cuTransA,
                                    static_cast<int>(N),
                                    static_cast<int>(M),
                                    static_cast<int>(K),
                                    &alpha,
                                    B,
                                    static_cast<int>(ldb),
                                    strideB,
                                    A,
                                    static_cast<int>(lda),
                                    strideA,
                                    &beta,
                                    C,
                                    static_cast<int>(ldc),
                                    strideC,
                                    static_cast<int>(batchCount));
    });

  }
}

template <>
template <typename T, typename U>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int64_t M,
                                        int64_t N,
                                        int64_t K,
                                        U alpha,
                                        const T *A,
                                        const T *B,
                                        U beta,
                                        T *C,
                                        int64_t batchCount,
                                        int64_t strideA,
                                        int64_t strideB) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  int64_t ldc = N;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  const int64_t strideC = M * N;
  
  if ((FLAGS_enable_cublas_tensor_op_math && (std::is_same<T, float>::value)) ||
      std::is_same<T, phi::float16>::value) {
    mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
    bool use_tensor_op_math = dev_ctx_.tensor_core_available();
    if (use_tensor_op_math) {
      algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
    VLOG(4) << "use_half_precision_compute_type: "
            << FLAGS_gemm_use_half_precision_compute_type;

    auto fp = std::is_same<T, float>::value ? MUSA_R_32F : MUSA_R_16F;
    auto compute_type = MUBLAS_COMPUTE_32F;

    float h_alpha = static_cast<float>(alpha);
    float h_beta = static_cast<float>(beta);
    void *a = static_cast<void *>(&h_alpha);
    void *b = static_cast<void *>(&h_beta);
    // set ComputeType as MUSA_R_32F for fp16, for better accuracy
    if (FLAGS_gemm_use_half_precision_compute_type == true &&
        std::is_same<T, phi::float16>::value) {
      a = static_cast<void *>(&alpha);
      b = static_cast<void *>(&beta);
      compute_type = MUBLAS_COMPUTE_16F;
    }

    if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE ||
        batchCount > INT_MAX_VALUE) {
#if !defined(PADDLE_WITH_MUSA)
      dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            phi::dynload::mublasGemmStridedBatchedEx_64(handle,
                                                        cuTransB,
                                                        cuTransA,
                                                        N,
                                                        M,
                                                        K,
                                                        a,
                                                        B,
                                                        fp,
                                                        ldb,
                                                        strideB,
                                                        A,
                                                        fp,
                                                        lda,
                                                        strideA,
                                                        b,
                                                        C,
                                                        fp,
                                                        ldc,
                                                        strideC,
                                                        batchCount,
                                                        compute_type,
                                                        algo));
      });
#else
      PADDLE_THROW(common::errors::Unimplemented(
          "mublasGemmStridedBatchedEx_64 is not supported on musa mp31"));
#endif  // !defined(PADDLE_WITH_MUSA)
    } else {
      dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
        PADDLE_ENFORCE_GPU_SUCCESS(phi::dynload::mublasGemmStridedBatchedEx(
            handle,
            cuTransB,
            cuTransA,
            static_cast<int>(N),
            static_cast<int>(M),
            static_cast<int>(K),
            a,
            B,
            fp,
            static_cast<int>(ldb),
            strideB,
            A,
            fp,
            static_cast<int>(lda),
            strideA,
            b,
            C,
            fp,
            static_cast<int>(ldc),
            strideC,
            static_cast<int>(batchCount),
            compute_type,
            algo));
      });
    }
  } else {
    T h_alpha = static_cast<T>(alpha);
    T h_beta = static_cast<T>(beta);
    dev_ctx_.CublasCall([&](mublasHandle_t handle) {
      CUBlas<T>::GEMM_STRIDED_BATCH(handle,
                                    cuTransB,
                                    cuTransA,
                                    static_cast<int>(N),
                                    static_cast<int>(M),
                                    static_cast<int>(K),
                                    &h_alpha,
                                    B,
                                    static_cast<int>(ldb),
                                    strideB,
                                    A,
                                    static_cast<int>(lda),
                                    strideA,
                                    &h_beta,
                                    C,
                                    static_cast<int>(ldc),
                                    strideC,
                                    static_cast<int>(batchCount));
    });

  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int64_t M,
                                               int64_t N,
                                               int64_t K,
                                               phi::bfloat16 alpha,
                                               const phi::bfloat16 *A,
                                               const phi::bfloat16 *B,
                                               phi::bfloat16 beta,
                                               phi::bfloat16 *C,
                                               int64_t batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int64_t lda = (transA == CblasNoTrans) ? K : M;
  int64_t ldb = (transB == CblasNoTrans) ? N : K;
  int64_t ldc = N;

  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  const int64_t strideC = M * N;

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

  mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
  bool use_tensor_op_math = dev_ctx_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE ||
      batchCount > INT_MAX_VALUE) {
#if !defined(PADDLE_WITH_MUSA)
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmStridedBatchedEx_64(handle,
                                                      cuTransB,
                                                      cuTransA,
                                                      N,
                                                      M,
                                                      K,
                                                      &h_alpha,
                                                      B,
                                                      MUSA_R_16BF,
                                                      ldb,
                                                      strideB,
                                                      A,
                                                      MUSA_R_16BF,
                                                      lda,
                                                      strideA,
                                                      &h_beta,
                                                      C,
                                                      MUSA_R_16BF,
                                                      ldc,
                                                      strideC,
                                                      batchCount,
                                                      MUBLAS_COMPUTE_32F,
                                                      algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmStridedBatchedEx_64 is not supported on musa mp31"));
#endif  // !defined(PADDLE_WITH_MUSA)
  } else {
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmStridedBatchedEx(handle,
                                                   cuTransB,
                                                   cuTransA,
                                                   static_cast<int>(N),
                                                   static_cast<int>(M),
                                                   static_cast<int>(K),
                                                   &h_alpha,
                                                   B,
                                                   MUSA_R_16BF,
                                                   static_cast<int>(ldb),
                                                   strideB,
                                                   A,
                                                   MUSA_R_16BF,
                                                   static_cast<int>(lda),
                                                   strideA,
                                                   &h_beta,
                                                   C,
                                                   MUSA_R_16BF,
                                                   static_cast<int>(ldc),
                                                   strideC,
                                                   static_cast<int>(batchCount),
                                                   MUBLAS_COMPUTE_32F,
                                                   algo));
    });
  }
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int64_t M,
                                               int64_t N,
                                               int64_t K,
                                               float alpha,
                                               const phi::bfloat16 *A,
                                               const phi::bfloat16 *B,
                                               float beta,
                                               phi::bfloat16 *C,
                                               int64_t batchCount,
                                               int64_t strideA,
                                               int64_t strideB) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  const int64_t strideC = M * N;

  float h_alpha = alpha;
  float h_beta = beta;

  mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
  bool use_tensor_op_math = dev_ctx_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");
  if (M > INT_MAX_VALUE || N > INT_MAX_VALUE || K > INT_MAX_VALUE ||
      batchCount > INT_MAX_VALUE) {
#if !defined(PADDLE_WITH_MUSA)
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmStridedBatchedEx_64(handle,
                                                      cuTransB,
                                                      cuTransA,
                                                      N,
                                                      M,
                                                      K,
                                                      &h_alpha,
                                                      B,
                                                      MUSA_R_16BF,
                                                      ldb,
                                                      strideB,
                                                      A,
                                                      MUSA_R_16BF,
                                                      lda,
                                                      strideA,
                                                      &h_beta,
                                                      C,
                                                      MUSA_R_16BF,
                                                      ldc,
                                                      strideC,
                                                      batchCount,
                                                      MUBLAS_COMPUTE_32F,
                                                      algo));
    });
#else
    PADDLE_THROW(common::errors::Unimplemented(
        "mublasGemmStridedBatchedEx_64 is not supported on musa mp31"));
#endif  // CUDA_VERSION >= 12030
  } else {
    dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
      PADDLE_ENFORCE_GPU_SUCCESS(
          phi::dynload::mublasGemmStridedBatchedEx(handle,
                                                   cuTransB,
                                                   cuTransA,
                                                   static_cast<int>(N),
                                                   static_cast<int>(M),
                                                   static_cast<int>(K),
                                                   &h_alpha,
                                                   B,
                                                   MUSA_R_16BF,
                                                   static_cast<int>(ldb),
                                                   strideB,
                                                   A,
                                                   MUSA_R_16BF,
                                                   static_cast<int>(lda),
                                                   strideA,
                                                   &h_beta,
                                                   C,
                                                   MUSA_R_16BF,
                                                   static_cast<int>(ldc),
                                                   strideC,
                                                   static_cast<int>(batchCount),
                                                   MUBLAS_COMPUTE_32F,
                                                   algo));
    });
  }
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                        CBLAS_TRANSPOSE transB,
                                        int M,
                                        int N,
                                        int K,
                                        T alpha,
                                        const T **A,
                                        const T **B,
                                        T beta,
                                        T **C,
                                        int batchCount) const {
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<T>(
        transA, transB, M, N, K, alpha, A[k], B[k], beta, C[k]);
  }
}

#if defined(__MUSACC__)
// TODO(jihong.zhong): below func need to fix 
// template <>
// template <>
// void Blas<phi::GPUContext>::BatchedGEMM<double>(CBLAS_TRANSPOSE transA,
//                                                CBLAS_TRANSPOSE transB,
//                                                int M,
//                                                int N,
//                                                int K,
//                                                double alpha,
//                                                const double **A,
//                                                const double **B,
//                                                double beta,
//                                                double **C,
//                                                int batchCount) const {
//   // Note that mublas follows fortran order, so the order is different from
//   // the cblas convention.
//   int lda = (transA == CblasNoTrans) ? K : M;
//   int ldb = (transB == CblasNoTrans) ? N : K;
//   int ldc = N;
//   mublasOperation_t cuTransA =
//       (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
//   mublasOperation_t cuTransB =
//       (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
//   thrust::device_vector<const double *> A_ptr(A, A + batchCount);
//   thrust::device_vector<const double *> B_ptr(B, B + batchCount);
//   thrust::device_vector<double *> C_ptr(C, C + batchCount);
// 
//   dev_ctx_.CublasCall([&](mublasHandle_t handle) {
//     CUBlas<double>::GEMM_BATCH(handle,
//                                cuTransB,
//                                cuTransA,
//                                N,
//                                M,
//                                K,
//                                &alpha,
//                                B_ptr.data().get(),
//                                ldb,
//                                A_ptr.data().get(),
//                                lda,
//                                &beta,
//                                C_ptr.data().get(),
//                                ldc,
//                                batchCount);
//   });
// }

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               float alpha,
                                               const float **A,
                                               const float **B,
                                               float beta,
                                               float **C,
                                               int batchCount) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  thrust::device_vector<const float *> A_ptr(A, A + batchCount);
  thrust::device_vector<const float *> B_ptr(B, B + batchCount);
  thrust::device_vector<float *> C_ptr(C, C + batchCount);

  dev_ctx_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<float>::GEMM_BATCH(handle,
                              cuTransB,
                              cuTransA,
                              N,
                              M,
                              K,
                              &alpha,
                              B_ptr.data().get(),
                              ldb,
                              A_ptr.data().get(),
                              lda,
                              &beta,
                              C_ptr.data().get(),
                              ldc,
                              batchCount);
  });
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::float16 alpha,
                                               const phi::float16 **A,
                                               const phi::float16 **B,
                                               phi::float16 beta,
                                               phi::float16 **C,
                                               int batchCount) const {
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      31,
      common::errors::InvalidArgument(
          "mublas fp16 gemm requires GPU compute capability >= 31,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));
  auto &cuda_ctx = const_cast<phi::GPUContext &>(dev_ctx_);
  CUBlas<phi::float16>::GEMM_BATCH(&cuda_ctx,
                                   cuTransB,
                                   cuTransA,
                                   N,
                                   M,
                                   K,
                                   &alpha,
                                   B,
                                   MUSA_R_16F,
                                   ldb,
                                   A,
                                   MUSA_R_16F,
                                   lda,
                                   &beta,
                                   C,
                                   MUSA_R_16F,
                                   ldc,
                                   batchCount,
                                   MUBLAS_COMPUTE_16F);
}

template <>
template <>
inline void Blas<phi::GPUContext>::BatchedGEMM(CBLAS_TRANSPOSE transA,
                                               CBLAS_TRANSPOSE transB,
                                               int M,
                                               int N,
                                               int K,
                                               phi::bfloat16 alpha,
                                               const phi::bfloat16 **A,
                                               const phi::bfloat16 **B,
                                               phi::bfloat16 beta,
                                               phi::bfloat16 **C,
                                               int batchCount) const {
#if defined(PADDLE_WITH_MUSA)
  // Note that mublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;

  PADDLE_ENFORCE_GE(
      dev_ctx_.GetComputeCapability(),
      31,
      common::errors::InvalidArgument(
          "mublas bf16 gemm requires GPU compute capability >= 80,"
          "but received %d",
          dev_ctx_.GetComputeCapability()));

  float f_alpha = static_cast<float>(alpha);
  float f_beta = static_cast<float>(beta);

  mublasGemmAlgo_t algo = MUBLAS_GEMM_DEFAULT;
  bool use_tensor_op_math = dev_ctx_.tensor_core_available();
  if (use_tensor_op_math) {
    algo = MUBLAS_GEMM_DEFAULT_TENSOR_OP;
  }
  VLOG(5) << "use_tensor_op_math: " << (use_tensor_op_math ? "True" : "False");

  thrust::device_vector<const void *> A_ptr(A, A + batchCount);
  thrust::device_vector<const void *> B_ptr(B, B + batchCount);
  thrust::device_vector<void *> C_ptr(C, C + batchCount);
  dev_ctx_.TensorCoreCublasCallIfAvailable([&](mublasHandle_t handle) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        phi::dynload::mublasGemmBatchedEx(handle,
                                          cuTransB,
                                          cuTransA,
                                          N,
                                          M,
                                          K,
                                          &f_alpha,
                                          B_ptr.data().get(),
                                          MUSA_R_16BF,
                                          ldb,
                                          A_ptr.data().get(),
                                          MUSA_R_16BF,
                                          lda,
                                          &f_beta,
                                          C_ptr.data().get(),
                                          MUSA_R_16BF,
                                          ldc,
                                          batchCount,
                                          MUBLAS_COMPUTE_32F,
                                          algo));
  });
#else
  // raise error
  PADDLE_THROW(common::errors::Unimplemented(
      "mublasGemmBatchedEx with bfloat16 is not supported on musa mp31"));

#endif  // CUDA_VERSION >= 11000
}
#endif

template <>
template <typename T>
void Blas<phi::GPUContext>::TRSM(CBLAS_SIDE side,
                                 CBLAS_UPLO uplo,
                                 CBLAS_TRANSPOSE transA,
                                 CBLAS_DIAG diag,
                                 int M,
                                 int N,
                                 T alpha,
                                 const T *A,
                                 int lda,
                                 T *B,
                                 int ldb) const {
  // solve row major `op ( A ) X = α B` by taking it as `X' op ( A' )  =  α B'`
  // where ' stands for transpose
  mublasSideMode_t cuSide =
      (side == CblasLeft) ? MUBLAS_SIDE_RIGHT : MUBLAS_SIDE_LEFT;
  mublasFillMode_t cuUplo =
      (uplo == CblasLower) ? MUBLAS_FILL_MODE_UPPER : MUBLAS_FILL_MODE_LOWER;
  // use MUBLAS_OP_C (conjugate transpose) for complex
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasDiagType_t cuDiag =
      (diag == CblasUnit) ? MUBLAS_DIAG_UNIT : MUBLAS_DIAG_NON_UNIT;

  dev_ctx_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::TRSM(
        handle, cuSide, cuUplo, cuTransA, cuDiag, N, M, &alpha, A, lda, B, ldb);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRF(
    int n, T **a, int *ipiv, int *info, int batch_size) const {
  dev_ctx_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::GETRF_BATCH(handle, n, a, n, ipiv, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRI(int n,
                                         const T **a,
                                         const int *ipiv,
                                         T **a_inv,
                                         int *info,
                                         int batch_size) const {
  PADDLE_ENFORCE_NE(
      a_inv,
      a,
      common::errors::InvalidArgument(
          "cuBLAS function 'mublas<S/D>getrfBatched' cannot be executed "
          "in-place. The memory space of output matrix (address: %p) cannot "
          "overlap memory space of input matrix (address: %p).",
          a_inv,
          a));
  dev_ctx_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::GETRI_BATCH(handle, n, a, n, ipiv, a_inv, n, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedMatInv(
    int n, const T **a, T **a_inv, int *info, int batch_size) const {
  dev_ctx_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::MATINV_BATCH(handle, n, a, n, a_inv, n, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedGETRS(CBLAS_TRANSPOSE trans,
                                         int n,
                                         int nrhs,
                                         const T **a,
                                         int lda,
                                         int *ipiv,
                                         T **b,
                                         int ldb,
                                         int *info,
                                         int batch_size) const {
  // use MUBLAS_OP_C (conjugate transpose) for complex
  mublasOperation_t cuTrans =
      (trans == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  dev_ctx_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::GETRS_BATCH(
        handle, cuTrans, n, nrhs, a, lda, ipiv, b, ldb, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<phi::GPUContext>::BatchedTRSM(CBLAS_SIDE side,
                                        CBLAS_UPLO uplo,
                                        CBLAS_TRANSPOSE transA,
                                        CBLAS_DIAG diag,
                                        int M,
                                        int N,
                                        T alpha,
                                        const T **A,
                                        int lda,
                                        T **B,
                                        int ldb,
                                        int batch_size) const {
  // solve row major `op ( A ) X = α B` by taking it as `X' op ( A' )  =  α B'`
  // where ' stands for transpose
  mublasSideMode_t cuSide =
      (side == CblasLeft) ? MUBLAS_SIDE_RIGHT : MUBLAS_SIDE_LEFT;
  mublasFillMode_t cuUplo =
      (uplo == CblasLower) ? MUBLAS_FILL_MODE_UPPER : MUBLAS_FILL_MODE_LOWER;
  // use MUBLAS_OP_C (conjugate transpose) for complex
  mublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? MUBLAS_OP_N : MUBLAS_OP_T;
  mublasDiagType_t cuDiag =
      (diag == CblasUnit) ? MUBLAS_DIAG_UNIT : MUBLAS_DIAG_NON_UNIT;

  dev_ctx_.CublasCall([&](mublasHandle_t handle) {
    CUBlas<T>::TRSM_BATCH(handle,
                          cuSide,
                          cuUplo,
                          cuTransA,
                          cuDiag,
                          N,
                          M,
                          &alpha,
                          A,
                          lda,
                          B,
                          ldb,
                          batch_size);
  });
}

}  // namespace funcs
}  // namespace phi
