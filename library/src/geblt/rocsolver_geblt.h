
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
#ifndef ROCSOLVER_GEBLT_H
#define ROCSOLVER_GEBLT_H

extern "C" {

rocblas_status rocsolver_dgeblttrf_interleaved_batch(rocblas_handle handle,
                                                     const rocblas_int nb,
                                                     const rocblas_int nblocks,
                                                     double* A,
                                                     lda,
                                                     double* B,
                                                     const rocblas_int ldb,
                                                     double* C,
                                                     const rocblas_int ldc,
                                                     const rocblas_int batchCount);

rocblas_status rocsolver_dgeblttrs_interleaved_batch(rocblas_handle handle,
                                                     const rocblas_int nb,
                                                     const rocblas_int nblocks,
                                                     double* A,
                                                     lda,
                                                     double* B,
                                                     const rocblas_int ldb,
                                                     double* C,
                                                     const rocblas_int ldc,
                                                     double* X_,
                                                     const rocblas_int ldx,
                                                     const rocblas_int batchCount);

rocblas_status rocsolver_dgeblttrf_strided_batched(rocblas_handle handle,
                                                   const rocblas_int nb,
                                                   const rocblas_int nblocks,
                                                   double* A,
                                                   const rocblas_int lda,
                                                   const rocblas_stride strideA,
                                                   double* B,
                                                   const rocblas_int ldb,
                                                   const rocblas_stride strideB,
                                                   double* C,
                                                   const rocblas_int ldc,
                                                   const rocblas_stride strideC,
                                                   rocblas_int devinfo_array[],
                                                   const rocblas_int batch_count);

rocblas_status rocsolver_dgeblttrs_strided_batched( rocblas_handle handle,
                                                    const rocblas_int nb,
};
#endif
