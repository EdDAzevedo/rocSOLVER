#pragma once
#ifndef ROCSOLVER_GTRF_BATCHED
#define ROCSOLVER_GTRF_BATCHED

#include "gbtr_common.h"
#include "gbtrf_npvt.hpp"

template <typename T>
GLOBAL_FUNCTION void gbtrf_npvt_batched_kernel(int nb,
                                               int nblocks,
                                               int batchCount,

                                               T* A_array[],
                                               int lda,
                                               T* B_array[],
                                               int ldb,
                                               T* C_array[],
                                               int ldc,
                                               int* pinfo)
{
    int SHARED_MEMORY sinfo;
#ifdef USE_GPU
    int const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    int const i_start = thread_id;
    int const i_inc = gridDim.x * blockDim.x;

    bool const is_root = (thread_id == 0);
    if(is_root)
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
            int linfo = 0;
            gbtrf_npvt_device<T>(nb, nblocks, A_array[i], lda, B_array[i], ldb, C_array[i], ldc,
                                 &linfo);
            info = max(info, linfo);
        };

        atomicMax(&sinfo, info);
        SYNCTHREADS;
    };

    if(is_root)
    {
        atomicMax(pinfo, sinfo);
    };
}

template <typename T>
rocblas_status gbtrf_npvt_batched_template(hipStream_t stream,
                                           int nb,
                                           int nblocks,
                                           int batchCount,

                                           T* A_array[],
                                           int lda,
                                           T* B_array[],
                                           int ldb,
                                           T* C_array[],
                                           int ldc,
                                           int* phost_info)
{
    *phost_info = 0;
    int* pdevice_info;
    HIP_CHECK(hipMalloc(&pdevice_info, sizeof(int)), rocblas_status_memory_error);
    HIP_CHECK(hipMemcpyHtoD(pdevice_info, phost_info, sizeof(int)), rocblas_status_internal_error);

    int grid_dim = (batchCount + (GBTR_BLOCK_DIM - 1)) / GBTR_BLOCK_DIM;
    hipLaunchKernelGGL((gbtrf_npvt_batched_kernel<T>), dim3(grid_dim), dim3(GBTR_BLOCK_DIM), 0,
                       stream,

                       nb, nblocks, batchCount,

                       A_array, lda, B_array, ldb, C_array, ldc,

                       pdevice_info);

    HIP_CHECK(hipMemcpyDtoH(phost_info, pdevice_info, sizeof(int)), rocblas_status_internal_error);
    HIP_CHECK(hipFree(pdevice_info), rocblas_status_memory_error);
    return (rocblas_status_success);
}

#endif
