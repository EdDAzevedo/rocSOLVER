
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
#ifndef ROCSOLVER_GTRF_STRIDED_BATCHED
#define ROCSOLVER_GTRF_STRIDED_BATCHED

#include "geblt_common.h"
#include "geblttrf_npvt.hpp"

template <typename T, typename I, typename Istride>
GLOBAL_FUNCTION void geblttrf_npvt_strided_batched_kernel(I nb,
                                                          I nblocks,
                                                          I batchCount,

                                                          T* A_,
                                                          I lda,
                                                          Istride strideA,
                                                          T* B_,
                                                          I ldb,
                                                          Istride strideB,
                                                          T* C_,
                                                          I ldc,
                                                          Istride strideC,
                                                          I devinfo_array[])
{
#ifdef USE_GPU
    auto const thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    auto const i_start = thread_id;
    auto const i_inc = gridDim.x * blockDim.x;

#else
    I const i_start = 0;
    I const i_inc = 1;
#endif

    {
        for(I i = i_start; i < batchCount; i += i_inc)
        {
            int64_t indxA = ((int64_t)strideA) * (i - 1);
            int64_t indxB = ((int64_t)strideB) * (i - 1);
            int64_t indxC = ((int64_t)strideC) * (i - 1);

            I linfo = 0;
            geblttrf_npvt_device<T>(nb, nblocks, &(A_[indxA]), lda, &(B_[indxB]), ldb, &(C_[indxC]),
                                    ldc, &linfo);

            devinfo_array[ i ] = linfo;
        };

        
    };

   
}

template <typename T, typename I, typename Istride>
rocblas_status geblttrf_npvt_strided_batched_template(hipStream_t stream,
                                                      I nb,
                                                      I nblocks,
                                                      I batchCount,

                                                      T* A_,
                                                      I lda,
                                                      Istride strideA,
                                                      T* B_,
                                                      I ldb,
                                                      Istride strideB,
                                                      T* C_,
                                                      I ldc,
                                                      Istride strideC,
                                                      I* phost_info)
{
    *phost_info = 0;
    I* pdevice_info;
    HIP_CHECK(hipMalloc(&pdevice_info, sizeof(I)), rocblas_status_memory_error);
    HIP_CHECK(hipMemcpyHtoD(pdevice_info, phost_info, sizeof(I)), rocblas_status_internal_error);

    I grid_dim = (batchCount + (GEBLT_BLOCK_DIM - 1)) / GEBLT_BLOCK_DIM;
    hipLaunchKernelGGL((geblttrf_npvt_strided_batched_kernel<T, I, Istride>), dim3(grid_dim),
                       dim3(GEBLT_BLOCK_DIM), 0, stream,

                       nb, nblocks, batchCount,

                       A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,

                       pdevice_info);

    HIP_CHECK(hipMemcpyDtoH(phost_info, pdevice_info, sizeof(I)), rocblas_status_internal_error);
    HIP_CHECK(hipFree(pdevice_info), rocblas_status_memory_error);
    return (rocblas_status_success);
}

#endif
