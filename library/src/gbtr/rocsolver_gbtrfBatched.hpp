
/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once
#ifndef ROCSOLVER_GTRF_BATCHED
#define ROCSOLVER_GTRF_BATCHED

#include "gbtr_common.h"
#include "gbtrf_npvt.hpp"

template <typename T, typename I>
GLOBAL_FUNCTION void gbtrf_npvt_batched_kernel(I nb,
                                               I nblocks,
                                               I batchCount,

                                               T* A_array[],
                                               I lda,
                                               T* B_array[],
                                               I ldb,
                                               T* C_array[],
                                               I ldc,
                                               I* pinfo)
{
    int SHARED_MEMORY sinfo;
#ifdef USE_GPU
    auto const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    auto const i_start = thread_id;
    auto const i_inc = gridDim.x * blockDim.x;

    bool const is_root = (thread_id == 0);
    if(is_root)
    {
        sinfo = 0;
    };
#else
    auto const i_start = 0;
    auto const i_inc = 1;
    sinfo = 0;
#endif

    {
        I info = 0;
        for(I i = i_start; i < batchCount; i += i_inc)
        {
            int linfo = 0;
            gbtrf_npvt_device<T,I>(nb, nblocks, A_array[i], lda, B_array[i], ldb, C_array[i], ldc,
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

template <typename T, typename I>
rocblas_status gbtrf_npvt_batched_template(hipStream_t stream,
                                           I nb,
                                           I nblocks,
                                           I batchCount,

                                           T* A_array[],
                                           I lda,
                                           T* B_array[],
                                           I ldb,
                                           T* C_array[],
                                           I ldc,
                                           I* phost_info)
{
    *phost_info = 0;
    I* pdevice_info;
    HIP_CHECK(hipMalloc(&pdevice_info, sizeof(I)), rocblas_status_memory_error);
    HIP_CHECK(hipMemcpyHtoD(pdevice_info, phost_info, sizeof(I)), rocblas_status_internal_error);

    auto grid_dim = (batchCount + (GBTR_BLOCK_DIM - 1)) / GBTR_BLOCK_DIM;
    hipLaunchKernelGGL((gbtrf_npvt_batched_kernel<T>), dim3(grid_dim), dim3(GBTR_BLOCK_DIM), 0,
                       stream,

                       nb, nblocks, batchCount,

                       A_array, lda, B_array, ldb, C_array, ldc,

                       pdevice_info);

    HIP_CHECK(hipMemcpyDtoH(phost_info, pdevice_info, sizeof(I)), rocblas_status_internal_error);
    HIP_CHECK(hipFree(pdevice_info), rocblas_status_memory_error);
    return (rocblas_status_success);
}

#endif
