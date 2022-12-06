/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSOLVER_CHECKARGS_GEBLT_NPVT_BATCHED_HPP
#define  ROCSOLVER_CHECKARGS_GEBLT_NPVT_BATCHED_HPP

#include "geblt_common.h"

template <typename T, typename I>
rocblas_status rocsolver_checkargs_geblt_npvt_batched(
                                               rocblas_handle handle,
                                               const I nb,
                                               const I nblocks,
                                               const I nrhs,

                                               T* A_array[],
                                               const I lda,
                                               T* B_array[],
                                               const I ldb,
                                               T* C_array[],
                                               const I ldc,

                                               T* X_array[],
                                               const I ldx,
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


    if((A_array == nullptr) || (B_array == nullptr) || (C_array == nullptr) || (X_array == nullptr))
    {
        return (rocblas_status_invalid_pointer);
    };


    bool const has_work = (nb >= 1) && (nblocks >= 1) && (batch_count >= 1) && (nrhs >= 1);

    if (has_work)
    {
        bool const isok =  (lda >= nb) && (ldb >= nb) && (ldc >= nb) && (ldx >= nb);
        if(!isok)
        {
            return (rocblas_status_invalid_size);
        };
     }

    // no work
    if(has_work) {
        return( rocblas_status_continue );
        }
    else {
        return (rocblas_status_success);
    };
};



#endif
