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
#ifndef ROCBLAS_CHECKARGS_GEBLT_NPVT_STRIDED_BATCHED_HPP
#define ROCBLAS_CHECKARGS_GEBLT_NPVT_STRIDED_BATCHED_HPP

#include <rocblas/rocblas.h>

template <typename T, typename I, typename Istride>
rocblas_status rocsolver_checkargs_geblt_npvt_strided_batched(rocblas_handle handle,
                                                              const I nb,
                                                              const I nblocks,
                                                              const I nrhs,
                                                              T* A_,
                                                              const I lda,
                                                              const Istride strideA,
                                                              T* B_,
                                                              const I ldb,
                                                              const Istride strideB,
                                                              T* C_,
                                                              const I ldc,
                                                              const Istride strideC,

                                                              T* X_,
                                                              const I ldx,
                                                              const Istride strideX,
                                                              const I batch_count)
{
    /* 
    ---------------
    check arguments
    ---------------
    */
    if(handle == nullptr)
    {
        return (rocblas_status_invalid_handle);
    };

    {
        bool const isok = (nb >= 0) && (nblocks >= 0) && (batch_count >= 0) && (strideA >= 0)
            && (strideB >= 0) && (strideC >= 0) && (strideX >= 0) && (lda >= nb) && (ldb >= nb)
            && (ldc >= nb) && (ldx >= nb);
        if(!isok)
        {
            return (rocblas_status_invalid_size);
        };

        // check no overlap
        bool const is_separated = (batch_count >= 2) && (strideA >= (lda * nb) * nblocks)
            && (strideB >= (ldb * nb) * nblocks) && (strideC >= (ldc * nb) * nblocks)
            && (strideX >= (ldx * nblocks) * nrhs);
        if(!is_separated)
        {
            return (rocblas_status_invalid_size);
        };
    };

    if((A_ == nullptr) || (B_ == nullptr) || (C_ == nullptr) || (X_ == nullptr))
    {
        return (rocblas_status_invalid_pointer);
    };

    {
        bool const has_work = (nb >= 1) && (nblocks >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        };
    };

    return (rocblas_status_continue);
};
#endif
