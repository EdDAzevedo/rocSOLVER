
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
#ifndef ROCSOLVER_GBTRF_INTERLEAVED_BATCH
#define ROCSOLVER_GBTRF_INTERLEAVED_BATCH

#include "gbtr_common.h"
#include "gbtrf_npvt_bf.hpp"

template <typename T, typename I>
rocblas_status rocsolver_gbtrfInterleavedBatch_template(rocblas_handle handle,
                                                        I nb,
                                                        I nblocks,
                                                        T* A_,
                                                        I lda,
                                                        T* B_,
                                                        I ldb,
                                                        T* C_,
                                                        I ldc,
                                                        I batchCount)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I info = 0;
    gbtrf_npvt_bf_template<T,I>(stream, nb, nblocks, batchCount, A_, lda, B_, ldb, C_, ldc, &info);

    return ((info == 0) ? rocblas_status_success : rocblas_status_internal_error);
}

#endif
