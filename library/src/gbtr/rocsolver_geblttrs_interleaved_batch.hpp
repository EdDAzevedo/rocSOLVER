
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
#ifndef ROCSOLVER_GBTRS_INTERLEAVED_BATCH
#define ROCSOLVER_GBTRS_INTERLEAVED_BATCH

#include "geblt_common.h"
#include "geblttrs_npvt_bf.hpp"

template <typename T, typename I>
rocblas_status rocsolver_geblttrs_interleaved_batch_template(rocblas_handle handle,
                                                             I nb,
                                                             I nblocks,
                                                             I nrhs,
                                                             const T* A_,
                                                             I lda,
                                                             const T* B_,
                                                             I ldb,
                                                             const T* C_,
                                                             I ldc,
                                                             T* brhs_,
                                                             I ldbrhs,
                                                             I batchCount)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I info = 0;

    geblttrs_npvt_bf_template<T>(stream, nb, nblocks, batchCount, nrhs, A_, lda, B_, ldb, C_, ldc,
                                 brhs_, ldbrhs, &info);

    return ((info == 0) ? rocblas_status_success : rocblas_status_internal_error);
}

#endif
