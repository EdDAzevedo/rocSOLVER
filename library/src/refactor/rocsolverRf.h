/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSOLVERRF_H
#define ROCSOLVERRF_H

#include "rocsolver_refactor.h"
// #include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#ifdef __cplusplus
extern "C" {
#endif

rocsolverStatus_t rocsolverRfAccessBundledFactors(/* Input */
                                                  rocsolverRfHandle_t handle,
                                                  /* Output (in the host memory ) */
                                                  int* nnzM,
                                                  /* Output (in the device memory) */
                                                  int** Mp,
                                                  int** Mi,
                                                  double** Mx);

rocsolverStatus_t rocsolverRfAnalyze(rocsolverRfHandle_t handle);

rocsolverStatus_t rocsolverRfCreate(rocsolverRfHandle_t* p_handle);

rocsolverStatus_t rocsolverRfRefactor(rocsolverRfHandle_t handle);

rocsolverStatus_t rocsolverRfResetValues(int n,
                                         int nnzA,
                                         int* csrRowPtrA,
                                         int* csrColIndA,
                                         double* csrValA,
                                         int* P,
                                         int* Q,

                                         rocsolverRfHandle_t handle);

rocsolverStatus_t rocsolverRfSetNumericProperties(rocsolverRfHandle_t handle,
                                                  double effective_zero,
                                                  double boost_val);

rocsolverStatus_t rocsolverRfSetupDevice(/* Input (in the device memory) */
                                         int n,
                                         int nnzA,
                                         int* csrRowPtrA,
                                         int* csrColIndA,
                                         double* csrValA,
                                         int nnzL,
                                         int* csrRowPtrL,
                                         int* csrColIndL,
                                         double* csrValL,
                                         int nnzU,
                                         int* csrRowPtrU,
                                         int* csrColIndU,
                                         double* csrValU,
                                         int* P,
                                         int* Q,

                                         /* Output */
                                         rocsolverRfHandle_t handle);

rocsolverStatus_t rocsolverRfSetupHost(/* Input (in the host memory) */
                                       int n,
                                       int nnzA,
                                       int* csrRowPtrA,
                                       int* csrColIndA,
                                       double* csrValA,
                                       int nnzL,
                                       int* csrRowPtrL,
                                       int* csrColIndL,
                                       double* csrValL,
                                       int nnzU,
                                       int* csrRowPtrU,
                                       int* csrColIndU,
                                       double* csrValU,
                                       int* P,
                                       int* Q,

                                       /* Output */
                                       rocsolverRfHandle_t handle);

rocsolverStatus_t rocsolverRfSetAlgs(rocsolverRfHandle_t handle,
                                     rocsolverRfFactorization_t fact_alg,
                                     rocsolverRfTriangularSolve_t alg);

/*
 ----------------------------
 interface for batch routines
 ----------------------------
*/

rocsolverStatus_t rocsolverRfBatchSetupHost(
    /* input in host memory */
    int batchSize,
    int n,

    int nnzA,
    int* h_csrRowPtrA,
    int* h_csrColIndA,
    double* h_csrValA_array[],

    int nnzL,
    int* h_csrRowPtrL,
    int* h_csrColIndL,
    double* h_csrValL,

    int nnzU,
    int* h_csrRowPtrU,
    int* h_csrColIndU,
    double* h_csrValU,

    int* h_P,
    int* h_Q,
    /* output */
    rocsolverRfHandle_t handle);

rocsolverStatus_t rocsolverRfBatchAnalyze( rocsolverRfHandle_t handle);

rocsolverStatus_t rocsolverRfBatchRefactor( rocsolverRfHandle_t handle);

rocsolverStatus_t rocsolverRfBatchResetValues(int batchSize,
                                           int n,
                                           int nnzA,

                                           /* input in device memory */

                                           int* csrRowPtrA,
                                           int* csrColIndA,
                                           double* csrValA_array[],
                                           int* P,
                                           int* Q,

                                           /* output */
                                           rocsolverRfHandle_t handle);

rocsolverStatus_t rocsolverRfBatchSolve(rocsolverRfHandle_t handle,
                                        int* d_P,
                                        int* d_Q,
                                        int nrhs,
                                        double* d_Temp,
                                        int ldt,
                                        double* d_XF_array[],
                                        int ldxf);

rocsolverStatus_t rocsolverRfBatchZeroPivot(rocsolverRfHandle_t handle,
                                            /* output in host memory */
                                            int* position);

#ifdef __cplusplus
};
#endif

#endif
