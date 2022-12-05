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
#ifndef ROCSOLVER_CHECKARGS_GEBLT_NPVT_INTERLEAVED_BATCH_HPP
#define ROCSOLVER_CHECKARGS_GEBLT_NPVT_INTERLEAVED_BATCH_HPP

#include "geblt_common.h"

template <typename T, typename I>
rocblas_status rocsolver_checkargs_geblt_npvt_interleaved_batch(rocblas_handle handle,
                                                                  I nb,
                                                                  I nblocks,
                                                                  T* A_,
                                                                  I lda,
                                                                  T* B_,
                                                                  I ldb,
                                                                  T* C_,
                                                                  I ldc,
                                                                  I devinfo_array[],
                                                                  I batch_count)
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


    bool const has_work = (nb >= 1) && (nblocks >= 1) && (batch_count >= 1);

    if((A_ == nullptr) || (B_ == nullptr) || (C_ == nullptr) || (devinfo_array == nullptr) )
    {
        return (rocblas_status_invalid_pointer);
    };
    {
        bool const isok = (nb >= 0) && (nblocks >= 0) && (batch_count >= 0) && 
                          (lda >= nb) && (ldb >= nb) && (ldc >= nb); 
        if(!isok)
        {
            return (rocblas_status_invalid_size);
        };
    };

    bool const no_work = !has_work;
    if (no_work) {
       return( rocblas_status_success );
       };

   return( rocblas_status_continue );
}
#endif
