

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

#include "rocsolver_gebltsv_npvt_batched.hpp"

extern "C" {

rocblas_status rocsolver_dgebltsv_npvt_batched(rocblas_handle handle,
                                               const rocblas_int nb,
                                               const rocblas_int nblocks,
                                               const rocblas_int nrhs,

                                               double* A_array[],
                                               const rocblas_int lda,

                                               double* B_array[],
                                               const rocblas_int ldb,

                                               double* C_array[],
                                               const rocblas_int ldc,

                                               double* X_array[],
                                               const rocblas_int ldx,

                                               rocblas_int info_array[],
                                               const rocblas_int batch_count)
{
    return (rocsolver_gebltsv_npvt_batched_impl(handle, nb, nblocks, nrhs, A_array, lda, B_array, ldb,
                                                C_array, ldc, X_array, ldx, info_array, batch_count));
};

rocblas_status rocsolver_sgebltsv_npvt_batched(rocblas_handle handle,
                                               const rocblas_int nb,
                                               const rocblas_int nblocks,
                                               const rocblas_int nrhs,

                                               float* A_array[],
                                               const rocblas_int lda,

                                               float* B_array[],
                                               const rocblas_int ldb,

                                               float* C_array[],
                                               const rocblas_int ldc,

                                               float* X_array[],
                                               const rocblas_int ldx,

                                               rocblas_int info_array[],
                                               const rocblas_int batch_count)
{
    return (rocsolver_gebltsv_npvt_batched_impl(handle, nb, nblocks, nrhs, A_array, lda, B_array, ldb,
                                                C_array, ldc, X_array, ldx, info_array, batch_count));
};

rocblas_status rocsolver_zgebltsv_npvt_batched(rocblas_handle handle,
                                               const rocblas_int nb,
                                               const rocblas_int nblocks,
                                               const rocblas_int nrhs,

                                               rocblas_double_complex* A_array[],
                                               const rocblas_int lda,

                                               rocblas_double_complex* B_array[],
                                               const rocblas_int ldb,

                                               rocblas_double_complex* C_array[],
                                               const rocblas_int ldc,

                                               rocblas_double_complex* X_array[],
                                               const rocblas_int ldx,

                                               rocblas_int info_array[],
                                               const rocblas_int batch_count)
{
    return (rocsolver_gebltsv_npvt_batched_impl(handle, nb, nblocks, nrhs, A_array, lda, B_array, ldb,
                                                C_array, ldc, X_array, ldx, info_array, batch_count));
};

rocblas_status rocsolver_cgebltsv_npvt_batched(rocblas_handle handle,
                                               const rocblas_int nb,
                                               const rocblas_int nblocks,
                                               const rocblas_int nrhs,

                                               rocblas_float_complex* A_array[],
                                               const rocblas_int lda,

                                               rocblas_float_complex* B_array[],
                                               const rocblas_int ldb,

                                               rocblas_float_complex* C_array[],
                                               const rocblas_int ldc,

                                               rocblas_float_complex* X_array[],
                                               const rocblas_int ldx,

                                               rocblas_int info_array[],
                                               const rocblas_int batch_count)
{
    return (rocsolver_gebltsv_npvt_batched_impl(handle, nb, nblocks, nrhs, A_array, lda, B_array, ldb,
                                                C_array, ldc, X_array, ldx, info_array, batch_count));
};
}
