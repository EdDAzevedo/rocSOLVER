
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
#include "rocsolver_gbtrsInterleavedBatch.hpp"

extern "C" {

rocblas_status rocsolverDgbtrsInterleavedBatch(rocblas_handle handle,
                                               rocblas_int nb,
                                               rocblas_int nblocks,
                                               rocblas_int nrhs,
                                               const double* A_,
                                               rocblas_int lda,
                                               const double* B_,
                                               rocblas_int ldb,
                                               const double* C_,
                                               rocblas_int ldc,
                                               double* brhs_,
                                               rocblas_int ldbrhs,
                                               rocblas_int batchCount)
{
    return (rocsolver_gbtrsInterleavedBatch_template<double, rocblas_int>(
        handle, nb, nblocks, nrhs, A_, lda, B_, ldb, C_, ldc, brhs_, ldbrhs, batchCount));
};

rocblas_status rocsolverSgbtrsInterleavedBatch(rocblas_handle handle,
                                               rocblas_int nb,
                                               rocblas_int nblocks,
                                               rocblas_int nrhs,
                                               const float* A_,
                                               rocblas_int lda,
                                               const float* B_,
                                               rocblas_int ldb,
                                               const float* C_,
                                               rocblas_int ldc,
                                               float* brhs_,
                                               rocblas_int ldbrhs,
                                               rocblas_int batchCount)
{
    return (rocsolver_gbtrsInterleavedBatch_template<float, rocblas_int>(
        handle, nb, nblocks, nrhs, A_, lda, B_, ldb, C_, ldc, brhs_, ldbrhs, batchCount));
};

rocblas_status rocsolverCgbtrsInterleavedBatch(rocblas_handle handle,
                                               rocblas_int nb,
                                               rocblas_int nblocks,
                                               rocblas_int nrhs,
                                               const rocblas_float_complex* A_,
                                               rocblas_int lda,
                                               const rocblas_float_complex* B_,
                                               rocblas_int ldb,
                                               const rocblas_float_complex* C_,
                                               rocblas_int ldc,
                                               rocblas_float_complex* brhs_,
                                               rocblas_int ldbrhs,
                                               rocblas_int batchCount)
{
    return (rocsolver_gbtrsInterleavedBatch_template<rocblas_float_complex>(
        handle, nb, nblocks, nrhs, A_, lda, B_, ldb, C_, ldc, brhs_, ldbrhs, batchCount));
};

rocblas_status rocsolverZgbtrsInterleavedBatch(rocblas_handle handle,
                                               rocblas_int nb,
                                               rocblas_int nblocks,
                                               rocblas_int nrhs,
                                               const rocblas_double_complex* A_,
                                               rocblas_int lda,
                                               const rocblas_double_complex* B_,
                                               rocblas_int ldb,
                                               const rocblas_double_complex* C_,
                                               rocblas_int ldc,
                                               rocblas_double_complex* brhs_,
                                               rocblas_int ldbrhs,
                                               rocblas_int batchCount)
{
    return (rocsolver_gbtrsInterleavedBatch_template<rocblas_double_complex, rocblas_int>(
        handle, nb, nblocks, nrhs, A_, lda, B_, ldb, C_, ldc, brhs_, ldbrhs, batchCount));
};
}
