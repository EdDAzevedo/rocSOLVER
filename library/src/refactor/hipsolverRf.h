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
#ifndef HIPSOLVERRF_H
#define HIPSOLVERRF_H

typedef void* hipsolverRfHandle_t;
typedef int hipsolverRfFactorization_t;
typedef int hipsolverRfTriangularSolve_t;

#ifdef __cplusplus
extern "C" {
#endif

rocsolverStatus_t hipsolverRfAccessBundledFactors(/* Input */
                                                  hipsolverRfHandle_t handle,
                                                  /* Output (in the host memory ) */
                                                  int* nnzM,
                                                  /* Output (in the device memory) */
                                                  int** Mp,
                                                  int** Mi,
                                                  double** Mx);

rocsolverStatus_t hipsolverRfAnalyze(hipsolverRfHandle_t handle);

rocsolverStatus_t hipsolverRfCreate(hipsolverRfHandle_t* p_handle);

rocsolverStatus_t hipsolverRfRefactor(hipsolverRfHandle_t handle);

rocsolverStatus_t hipsolverRfResetValues(int n,
                                         int nnzA,
                                         int* csrRowPtrA,
                                         int* csrColIndA,
                                         double* csrValA,
                                         int* P,
                                         int* Q,

                                         hipsolverRfHandle_t handle);

rocsolverStatus_t hipsolverRfSetNumericProperties(hipsolverRfHandle_t handle,
                                                  double effective_zero,
                                                  double boost_val);

rocsolverStatus_t hipsolverRfSetupDevice(/* Input (in the device memory) */
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
                                         hipsolverRfHandle_t handle);

rocsolverStatus_t hipsolverRfSetupHost(/* Input (in the host memory) */
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
                                       hipsolverRfHandle_t handle);

rocsolverStatus_t hipsolverRfSetAlgs(hipsolverRfHandle_t handle,
                                     hipsolverRfFactorization_t fact_alg,
                                     hipsolverRfTriangularSolve_t alg);

/*
 ----------------------------
 interface for batch routines
 ----------------------------
*/

hipsolverStatus_t hipsolverRfBatchSetupHost(
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
    hipsolverRfHandle_t handle);

hipsolverStatus_t hipsolverRfBatchAnalyze();

hipsolverStatus_t hipsolverRfBatchResetValues(hipsolverRfHandle_t handle);

hipsolverStatus_t hipsolverRfBatchRefactor(int batchSize,
                                           int n,
                                           int nnzA,

                                           /* input in device memory */

                                           int* csrRowPtrA,
                                           int* csrColIndA,
                                           double* csrValA_array[],
                                           int* P,
                                           int* Q,

                                           /* output */
                                           hipsolverRfHandle_t handle);

hipsolverStatus_t hipsolverRfBatchSolve(hipsolverRfHandle_t handle,
                                        int* d_P,
                                        int* d_Q,
                                        int nrhs,
                                        double* d_Temp,
                                        int ldt,
                                        double* d_XF_array[],
                                        int ldxf);

hipsolverStatus_t hipsolverRfBatchZeroPivot(hipsolverRfHandle_t handle,
                                            /* output in host memory */
                                            int* position);

#ifdef __cplusplus
};
#endif

#endif
