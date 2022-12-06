
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

#include "rocsolver_checkargs_geblt_npvt_strided_batched.hpp"
#include "rocsolver_geblttrf_npvt_strided_batched_large.hpp"
#include "rocsolver_geblttrf_npvt_strided_batched_small.hpp"

template <typename T, typename I, typename Istride>
rocblas_status rocsolver_geblttrf_npvt_strided_batched_impl(rocblas_handle handle,
                                                            I nb,
                                                            I nblocks,
                                                            T* A_,
                                                            I lda,
                                                            Istride strideA,
                                                            T* B_,
                                                            I ldb,
                                                            Istride strideB,
                                                            T* C_,
                                                            I ldc,
                                                            Istride strideC,
                                                            I* devinfo_array,
                                                            I batch_count)
{
    {
        ROCSOLVER_ENTER_TOP("getrf_npvt_strided_batched", "-nb", nb, "-nblocks", nblocks, "--lda",
                            lda, "--strideA", strideA, "--strideB", strideB, "--batch_count",
                            batch_count);
    };

    {
        // ---------------
        // check arguments
        // ---------------

        // ------------------------------------------------------------
        // reuse  values of A_,lda, strideA, for dummy X_, ldx, strideX
        // ------------------------------------------------------------
        T* X_ = A_;
        const I ldx = lda;
        const Istride strideX = strideA;
        const I nrhs = nb;

        rocblas_status istat = rocsolver_checkargs_geblt_npvt_strided_batched(
            handle, nb, nblocks, nrhs, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC, X_,
            ldx, strideX, batch_count);

        if(istat != rocblas_status_continue)
        {
            return (istat);
        };

        if(devinfo_array == nullptr)
        {
            return (rocblas_status_invalid_pointer);
        };
    }

    // ----------------------------
    // set devinfo_array to be zero
    // ----------------------------
    if(batch_count >= 1)
    {
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);

        void* dst = (void*)&(devinfo_array[0]);
        int value = 0;
        size_t sizeBytes = sizeof(I) * batch_count;

        HIP_CHECK(hipMemsetAsync(dst, value, sizeBytes, stream), rocblas_status_internal_error);
    };

    if(nb < NB_SMALL)
    {
        return (rocsolver_geblttrf_npvt_strided_batched_small_template(
            handle, nb, nblocks, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,
            devinfo_array, batch_count));
    }
    else
    {
        return (rocsolver_geblttrf_npvt_strided_batched_large_template(
            handle, nb, nblocks, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,
            devinfo_array, batch_count));
    };
};
