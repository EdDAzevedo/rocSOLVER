#pragma once
#ifndef ROCSOLVER_GTRF_STRIDED_BATCH
#define ROCSOLVER_GTRF_STRIDED_BATCH

#include "gbtr_common.h"
#include "gbtrf_npvt.hpp"

template <typename T>
GLOBAL_FUNCTION void gbtr_npvt_strided_batched_kernel(int nb,
                                                      int nblocks,
                                                      int batchCount,

                                                      T* A_,
                                                      int lda,
                                                      rocblas_stride strideA,
                                                      T* B_,
                                                      int ldb,
                                                      rocblas_stride strideB,
                                                      T* C_,
                                                      int ldc,
                                                      rocblas_stride strideC,
                                                      int* pinfo)
{
    int SHARED_MEMORY sinfo;
#ifdef USE_GPU
    int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int const i_start = thread_id;
    int const i_inc = gridDim.x * blockDim.x;

    if(thread_id == 0)
    {
        sinfo = 0;
    };
#else
    int const i_start = 0;
    int const i_inc = 1;
    sinfo = 0;
#endif

    {
        int info = 0;
        for(int i = i_start; i < batchCount; i += i_inc)
        {
            int64_t indxA = ((int64_t)strideA) * (i - 1);
            int64_t indxB = ((int64_t)strideB) * (i - 1);
            int64_t indxC = ((int64_t)strideC) * (i - 1);

            int linfo = 0;
            gbtrf_npvt<T>(nb, nblocks, A_[indxA], lda, B_[indxB], ldb, C_[indxC], ldc, &linfo);
            info = max(info, linfo);
        };

        SYNCTHREADS;
        atomicMax(&sinfo, info);
        SYNCTHREADS;
    };

    atomicMax(pinfo, sinfo);
}

template <typename T>
rocblas_status gbtrf_npvt_strided_batched_template(hipStream_t stream,
                                                   int nb,
                                                   int nblocks,
                                                   int batchCount,

                                                   T* A_,
                                                   int lda,
                                                   rocblas_stride strideA,
                                                   T* B_,
                                                   int ldb,
                                                   rocblas_stride strideB,
                                                   T* C_,
                                                   int ldc,
                                                   rocblas_stride strideC,
                                                   int* phost_info)
{
    *phost_info = 0;
    int* pdevice_info;
    HIP_CHECK(hipMalloc(&pdevice_info, sizeof(int)), rocblas_status_memory_error);
    HIP_CHECK(hipMemcpyHtoD(pdevice_info, phost_info, sizeof(int)), rocblas_status_internal_error);

    int grid_dim = (batchCount + (GBTR_BLOCK_DIM - 1)) / GBTR_BLOCK_DIM;
    hipLaunchKernelGGL((gbtrf_npvt_strided_batched_template<T>), dim3(grid_dim), dim3(GBTR_BLOCK_DIM), 0,
                       stream,

                       nb, nblocks, batchCount,

                       A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,

                       pdevice_info);

    HIP_CHECK(hipMemcpyDtoH(phost_info, pdevice_info, sizeof(int)), rocblas_status_internal_error);
    HIP_CHECK(hipFree(pdevice_info), rocblas_status_memory_error);
    return (rocblas_status_success);
}

#endif
