
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
#include "gbtrf_npvt_bf.hpp"
#include "gbtr_common.h"

extern "C" {

void Dgbtrf_npvt_bf(hipStream_t stream,

                    int const nb,
                    int const nblocks,
                    int const batchCount,
                    double* A_,
                    int const lda,
                    double* B_,
                    int const ldb,
                    double* C_,
                    int const ldc,
                    int* pinfo)
{
    gbtrf_npvt_bf_template<double>(stream, nb, nblocks, batchCount, A_, lda, B_, ldb, C_, ldc, pinfo);
};

void Zgbtrf_npvt_bf(hipStream_t stream,
                    int const nb,
                    int const nblocks,
                    int const batchCount,
                    rocblas_double_complex* A_,
                    int const lda,
                    rocblas_double_complex* B_,
                    int const ldb,
                    rocblas_double_complex* C_,
                    int const ldc,
                    int* pinfo)
{
    gbtrf_npvt_bf_template<rocblas_double_complex>(stream, nb, nblocks, batchCount, A_, lda, B_,
                                                   ldb, C_, ldc, pinfo);
};
}
