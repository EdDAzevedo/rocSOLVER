
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
#ifndef ROCSOLVER_GBTR_H
#define ROCSOLVER_GBTR_H

extern "C" {

rocblas_status rocsolverDgbtrfInterleavedBatch(rocblas_handle handle,
                                               int nb,
                                               int nblocks,
                                               const double* A_,
                                               int lda,
                                               const double* B_,
                                               int ldb,
                                               const double* C_,
                                               int ldc,
                                               int batchCount);

rocblas_status rocsolverSgbtrfInterleavedBatch(rocblas_handle handle,
                                               int nb,
                                               int nblocks,
                                               const float* A_,
                                               int lda,
                                               const float* B_,
                                               int ldb,
                                               const float* C_,
                                               int ldc,
                                               int batchCount);

rocblas_status rocsolverDgbtrsInterleavedBatch(rocsolverHandlt_t handle,
                                               int nb,
                                               int nblocks,
                                               const double* A_,
                                               int lda,
                                               const double* B_,
                                               int ldb,
                                               const double* C_,
                                               int ldc,
                                               double* brhs_,
                                               int ldbrhs,
                                               int batchCount);

rocblas_status rocsolverSgbtrsInterleavedBatch(rocsolverHandlt_t handle,
                                               int nb,
                                               int nblocks,
                                               const float* A_,
                                               int lda,
                                               const float* B_,
                                               int ldb,
                                               const float* C_,
                                               int ldc,
                                               float* brhs_,
                                               int ldbrhs,
                                               int batchCount);

rocblas_status rocsolverDgbtrfStridedBatch(rocblas_handle handle,
                                           int nb,
                                           int nblocks,
                                           double* A_,
                                           int lda,
                                           int strideA,
                                           double* B_,
                                           int ldb,
                                           int strideB,
                                           double* C_,
                                           int ldc,
                                           int strideC,
                                           int batchCount);
};
#endif
