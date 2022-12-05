
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
#include "rocsolver_geblttrf_npvt_interleaved_batch.hpp"

extern "C" {

rocblas_status rocsolver_dgeblttrf_npvt_interleaved_batch(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          double* A_,
                                                          rocblas_int lda,
                                                          double* B_,
                                                          rocblas_int ldb,
                                                          double* C_,
                                                          rocblas_int ldc,
                                                          rocblas_int devinfo_array[],
                                                          rocblas_int batchCount)
{
    return (rocsolver_geblttrf_npvt_interleaved_batch_impl<double, rocblas_int>(
        handle, nb, nblocks, A_, lda, B_, ldb, C_, ldc, devinfo_array, batchCount));
};

rocblas_status rocsolver_sgeblttrf_npvt_interleaved_batch(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          float* A_,
                                                          rocblas_int lda,
                                                          float* B_,
                                                          rocblas_int ldb,
                                                          float* C_,
                                                          rocblas_int ldc,
                                                          rocblas_int devinfo_array[],
                                                          rocblas_int batchCount)
{
    return (rocsolver_geblttrf_npvt_interleaved_batch_impl<float, rocblas_int>(
        handle, nb, nblocks, A_, lda, B_, ldb, C_, ldc, devinfo_array, batchCount));
};

rocblas_status rocsolver_zgeblttrf_npvt_interleaved_batch(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          rocblas_double_complex* A_,
                                                          rocblas_int lda,
                                                          rocblas_double_complex* B_,
                                                          rocblas_int ldb,
                                                          rocblas_double_complex* C_,
                                                          rocblas_int ldc,
                                                          rocblas_int devinfo_array[],
                                                          rocblas_int batchCount)
{
    return (rocsolver_geblttrf_npvt_interleaved_batch_impl<rocblas_double_complex, rocblas_int>(
        handle, nb, nblocks, A_, lda, B_, ldb, C_, ldc, devinfo_array, batchCount));
};

rocblas_status rocsolver_cgeblttrf_npvt_interleaved_batch(rocblas_handle handle,
                                                          rocblas_int nb,
                                                          rocblas_int nblocks,
                                                          rocblas_float_complex* A_,
                                                          rocblas_int lda,
                                                          rocblas_float_complex* B_,
                                                          rocblas_int ldb,
                                                          rocblas_float_complex* C_,
                                                          rocblas_int ldc,
                                                          rocblas_int devinfo_array[],
                                                          rocblas_int batchCount)
{
    return (rocsolver_geblttrf_npvt_interleaved_batch_impl<rocblas_float_complex, rocblas_int>(
        handle, nb, nblocks, A_, lda, B_, ldb, C_, ldc, devinfo_array, batchCount));
};
}
