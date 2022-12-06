

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

#include "rocsolver_gebltsv_npvt_strided_batched.hpp"

extern "C" {

rocblas_status rocsolver_dgebltsv_npvt_strided_batched(rocblas_handle handle,
                                                       const rocblas_int nb,
                                                       const rocblas_int nblocks,
                                                       const rocblas_int nrhs,

                                                       double* A_,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,

                                                       double* B_,
                                                       const rocblas_int ldb,
                                                       const rocblas_stride strideB,

                                                       double* C_,
                                                       const rocblas_int ldc,
                                                       const rocblas_stride strideC,

                                                       double* X_,
                                                       const rocblas_int ldx,
                                                       const rocblas_stride strideX,

                                                       rocblas_int info_array[],
                                                       const rocblas_int batch_count)
{
    return (rocsolver_gebltsv_npvt_strided_batched_impl(handle, nb, nblocks, nrhs, A_, lda, strideA,
                                                        B_, ldb, strideB, C_, ldc, strideC, X_, ldx,
                                                        strideX, info_array, batch_count));
};

rocblas_status rocsolver_sgebltsv_npvt_strided_batched(rocblas_handle handle,
                                                       const rocblas_int nb,
                                                       const rocblas_int nblocks,
                                                       const rocblas_int nrhs,

                                                       float* A_,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,

                                                       float* B_,
                                                       const rocblas_int ldb,
                                                       const rocblas_stride strideB,

                                                       float* C_,
                                                       const rocblas_int ldc,
                                                       const rocblas_stride strideC,

                                                       float* X_,
                                                       const rocblas_int ldx,
                                                       const rocblas_stride strideX,

                                                       rocblas_int info_array[],
                                                       const rocblas_int batch_count)
{
    return (rocsolver_gebltsv_npvt_strided_batched_impl(handle, nb, nblocks, nrhs, A_, lda, strideA,
                                                        B_, ldb, strideB, C_, ldc, strideC, X_, ldx,
                                                        strideX, info_array, batch_count));
};

rocblas_status rocsolver_zgebltsv_npvt_strided_batched(rocblas_handle handle,
                                                       const rocblas_int nb,
                                                       const rocblas_int nblocks,
                                                       const rocblas_int nrhs,

                                                       rocblas_double_complex* A_,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,

                                                       rocblas_double_complex* B_,
                                                       const rocblas_int ldb,
                                                       const rocblas_stride strideB,

                                                       rocblas_double_complex* C_,
                                                       const rocblas_int ldc,
                                                       const rocblas_stride strideC,

                                                       rocblas_double_complex* X_,
                                                       const rocblas_int ldx,
                                                       const rocblas_stride strideX,

                                                       rocblas_int info_array[],
                                                       const rocblas_int batch_count)
{
    return (rocsolver_gebltsv_npvt_strided_batched_impl(handle, nb, nblocks, nrhs, A_, lda, strideA,
                                                        B_, ldb, strideB, C_, ldc, strideC, X_, ldx,
                                                        strideX, info_array, batch_count));
};

rocblas_status rocsolver_cgebltsv_npvt_strided_batched(rocblas_handle handle,
                                                       const rocblas_int nb,
                                                       const rocblas_int nblocks,
                                                       const rocblas_int nrhs,

                                                       rocblas_float_complex* A_,
                                                       const rocblas_int lda,
                                                       const rocblas_stride strideA,

                                                       rocblas_float_complex* B_,
                                                       const rocblas_int ldb,
                                                       const rocblas_stride strideB,

                                                       rocblas_float_complex* C_,
                                                       const rocblas_int ldc,
                                                       const rocblas_stride strideC,

                                                       rocblas_float_complex* X_,
                                                       const rocblas_int ldx,
                                                       const rocblas_stride strideX,

                                                       rocblas_int info_array[],
                                                       const rocblas_int batch_count)
{
    return (rocsolver_gebltsv_npvt_strided_batched_impl(handle, nb, nblocks, nrhs, A_, lda, strideA,
                                                        B_, ldb, strideB, C_, ldc, strideC, X_, ldx,
                                                        strideX, info_array, batch_count));
};
}
