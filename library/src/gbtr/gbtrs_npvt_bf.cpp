
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
#include "gbtrs_npvt_bf.hpp"
#include "gbtr_common.h"

extern "C" {

void Dgbtrs_npvt_bf(hipStream_t stream,

                    int const nb,
                    int const nblocks,
                    int const batchCount,
                    int const nrhs,
                    double const* const A_,
                    int const lda,
                    double const* const D_,
                    int const ldd,
                    double const* const U_,
                    int const ldu,
                    double* brhs_,
                    int const ldbrhs,
                    int* pinfo)
{
    gbtrs_npvt_bf_template<double>(stream,

                                   nb, nblocks, batchCount, nrhs, A_, lda, D_, ldd, U_, ldu, brhs_,
                                   ldbrhs, pinfo);
};

void Zgbtrs_npvt_bf(hipStream_t stream,

                    int const nb,
                    int const nblocks,
                    int const batchCount,
                    int const nrhs,
                    rocblas_double_complex const* const A_,
                    int const lda,
                    rocblas_double_complex const* const D_,
                    int const ldd,
                    rocblas_double_complex const* const U_,
                    int const ldu,
                    rocblas_double_complex* brhs_,
                    int const ldbrhs,
                    int* pinfo)
{
    gbtrs_npvt_bf_template<rocblas_double_complex>(stream,

                                                   nb, nblocks, batchCount, nrhs, A_, lda, D_, ldd,
                                                   U_, ldu, brhs_, ldbrhs, pinfo);
};
}
