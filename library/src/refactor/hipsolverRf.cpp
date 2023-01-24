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

#include "hipsolverRf.h"

#ifdef __cplusplus
extern "C" {
#endif

hipsolverStatus_t hipsolverRfAccessBundledFactors(/* Input */
                                                  hipsolverRfHandle_t handle,
                                                  /* Output (in the host memory ) */
                                                  int* nnzM,
                                                  /* Output (in the device memory) */
                                                  int** Mp,
                                                  int** Mi,
                                                  double** Mx)
{
    return ((hipsolverStatus_t)rocsolverRfAccessBundledFactors((rocsolverRfHandle_t)handle, nnzM,
                                                               Mp, Mi, Mx));
};

hipsolverStatus_t hipsolverRfAnalyze(hipsolverRfHandle_t handle)
{
    return ((hipsolverStatus_t)rocsolverRfAnalyze((rocsolverRfHandle_t)handle));
};

hipsolverStatus_t hipsolverRfCreate(hipsolverRfHandle_t* p_handle)
{
    return ((hipsolverStatus_t)rocsolverRfCreate((rocsolverRfHandle_t*)p_handle));
};

hipsolverStatus_t hipsolverRfExtractBundledFactorsHost(hipsolverRfHandle_t handle,
                                                       int* h_nnzM,
                                                       int** h_Mp,
                                                       int** h_Mi,
                                                       double** h_Mx)
{
    return ((hipsolverStatus_t)rocsolverRfExtractBundledFactorsHost((rocsolverRfHandle_t)handle,
                                                                    h_nnzM, h_Mp, h_Mi, h_Mx));
};

hipsolverStatus_t hipsolverRfExtractSplitFactorsHost(hipsolverRfHandle_t handle,
                                                     int* h_nnzL,
                                                     int** h_Lp,
                                                     int** h_Li,
                                                     double** h_Lx,
                                                     int* h_nnzU,
                                                     int** h_Up,
                                                     int** h_Ui,
                                                     double** h_Ux)
{
    return ((hipsolverStatus_t)rocsolverRfExtractSplitFactorsHost(
        (rocsolverRfHandle_t)handle, h_nnzL, h_Lp, h_Li, h_Lx, h_nnzU, h_Up, h_Ui, h_Ux));
};

hipsolverStatus_t hipsolverRfDestroy(hipsolverRfHandle_t handle)
{
    return ((hipsolverStatus_t)rocsolverRfDestroy((rocsolverRfHandle_t)handle));
};

hipsolverStatus_t hipsolverRfGetMatrixFormat(hipsolverRfHandle_t handle,
                                             hipsolverRfMatrixFormat_t* format,
                                             hipsolverRfUnitDiagonal_t* diag)
{
    return ((hipsolverStatus_t)rocsolverRfGetMatrixFormat((rocsolverRfHandle_t)handle,
                                                          (rocsolverRfMatrixFormat_t*)format,
                                                          (rocsolverRfUnitDiagonal_t*)diag));
};

hipsolverStatus_t
    hipsolverRfGetNumericProperties(hipsolverRfHandle_t handle, double* zero, double* boost)
{
    return ((hipsolverStatus_t)rocsolverRfGetNumericProperties(handle, zero, boost));
};

hipsolverStatus_t hipsolverRfGetNumericBoostReport(hipsolverRfHandle_t handle,
                                                   hipsolverRfNumericBoostReport_t* report)
{
    return ((hipsolverStatus_t)rocsolverRfGetNumericBoostReport(
        (rocsolverRfHandle_t)handle, (rocsolverRfNumericBoostReport_t*)report));
};

hipsolverStatus_t hipsolverRfGetResetValuesFastMode(hipsolverRfHandle_t handle,
                                                    hipsolverRfResetValuesFastMode_t* fastMode)
{
    return ((hipsolverStatus_t)rocsolverRfGetResetValuesFastMode(
        (rocsolverRfHandle_t)handle, (rocsolverRfResetValuesFastMode_t*)fastMode));
};

hipsolverStatus_t hipsolverRfGet_Algs(hipsolverRfHandle_t handle,
                                      hipsolverRfFactorization_t* fact_alg,
                                      hipsolverRfTriangularSolve_t* solve_alg)
{
    return ((hipsolverStatus_t)rocsolverRfGet_Algs((rocsolverRfHandle_t)handle,
                                                   (rocsolverRfFactorization_t*)fact_alg,
                                                   (rocsolverRfTriangularSolve_t*)solve_alg));
};

hipsolverStatus_t hipsolverRfRefactor(hipsolverRfHandle_t handle)
{
    return ((hipsolverStatus_t)rocsolverRfRefactor((rocsolverRfHandle_t)handle));
};

hipsolverStatus_t hipsolverRfResetValues(int n,
                                         int nnzA,
                                         int* csrRowPtrA,
                                         int* csrColIndA,
                                         double* csrValA,
                                         int* P,
                                         int* Q,

                                         hipsolverRfHandle_t handle)
{
    return ((hipsolverStatus_t)rocsolverRfResetValues(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P,
                                                      Q, (rocsolverRfHandle_t)handle));
};

hipsolverStatus_t hipsolverRfSetMatrixFormat(hipsolverRfHandle_t handle,
                                             hipsolverRfMatrixFormat_t format,
                                             hipsolverRfUnitDiagonal_t diag)
{
    return ((hipsolverStatus_t)rocsolverRfSetMatrixFormat(
        (rocsolverRfHandle_t)handle, (gluMatrixFormat_t)format, (gluUnitDiagonal_t)diag));
};

hipsolverStatus_t hipsolverRfSetNumericProperties(hipsolverRfHandle_t handle,
                                                  double effective_zero,
                                                  double boost_val)
{
    return ((hipsolverStatus_t)rocsolverRfSetNumericProperties((rocsolverRfHandle_t)handle,
                                                               effective_zero, boost_val));
};

hipsolverStatus_t hipsolverRfSetResetValuesFastMode(hipsolverRfHandle_t handle,
                                                    hipsolverRfResetValuesFastMode_t fastMode)
{
    return ((hipsolverStatus_t)rocsolverRfSetResetValuesFastMode(
        (rocsolverRfHandle_t)handle, (gluResetValuesFastMode_t)fastMode));
};

hipsolverStatus_t hipsolverRfSetupDevice(/* Input (in the device memory) */
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
                                         hipsolverRfHandle_t handle)
{
    return ((hipsolverStatus_t)rocsolverRfSetupDevice(
        n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU,
        csrRowPtrU, csrColIndU, csrValU, P, Q, (rocsolverRfHandle_t)handle));
};

hipsolverStatus_t hipsolverRfSetupHost(/* Input (in the host memory) */
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
                                       hipsolverRfHandle_t handle)
{
    return ((hipsolverStatus_t)rocsolverRfSetupHost(
        n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, csrColIndL, csrValL, nnzU,
        csrRowPtrU, csrColIndU, csrValU, P, Q, (rocsolverRfHandle_t)handle));
};

hipsolverStatus_t hipsolverRfSetAlgs(hipsolverRfHandle_t handle,
                                     hipsolverRfFactorization_t fact_alg,
                                     hipsolverRfTriangularSolve_t alg)
{
    return ((hipsolverStatus_t)rocsolverRfSetAlgs((rocsolverRfHandle_t)handle,
                                                  (rocsolverRfFactorization_t)fact_alg,
                                                  (rocsolverRfTriangularSolve_t)alg));
};

hipsolverStatus_t hipsolverRfSolve(hipsolverRfHandle_t handle,
                                   int* P,
                                   int* Q,
                                   int nrhs,
                                   double* Temp,
                                   int ldt,
                                   double* XF,
                                   int ldxf)
{
    return ((hipsolverStatus_t)rocsolverRfSolve((rocsolverRfHandle_t)handle, P, Q, nrhs, Temp, ldt,
                                                XF, ldxf));
};

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
    hipsolverRfHandle_t handle)
{
    return ((hipsolverStatus_t)rocsolverRfBatchSetupHost(
        batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA_array, nnzL, h_csrRowPtrL,
        h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q,
        (rocsolverRfHandle_t)handle));
};

hipsolverStatus_t hipsolverRfBatchAnalyze(hipsolverRfHandle_t handle)
{
    return ((hipsolverStatus_t)rocsolverRfBatchAnalyze((rocsolverRfHandle_t)handle));
};

hipsolverStatus_t hipsolverRfBatchResetValues(int batchSize,
                                              int n,
                                              int nnzA,
                                              int* csrRowPtrA,
                                              int* csrColIndA,
                                              double* csrValA_array[],
                                              int* P,
                                              int* Q,
                                              hipsolverRfHandle_t handle)
{
    return ((hipsolverStatus_t)rocsolverRfBatchResetValues(batchSize, n, nnzA, csrRowPtrA,
                                                           csrColIndA, csrValA_array, P, Q,
                                                           (rocsolverRfHandle_t)handle));
};

hipsolverStatus_t hipsolverRfBatchRefactor(hipsolverRfHandle_t handle)
{
    return ((hipsolverStatus_t)rocsolverRfBatchRefactor((rocsolverRfHandle_t)handle));
};

hipsolverStatus_t hipsolverRfBatchSolve(hipsolverRfHandle_t handle,
                                        int* d_P,
                                        int* d_Q,
                                        int nrhs,
                                        double* d_Temp,
                                        int ldt,
                                        double* d_XF_array[],
                                        int ldxf)
{
    return ((hipsolverStatus_t)rocsolverRfBatchSolve((rocsolverRfHandle_t)handle, d_P, d_Q, nrhs,
                                                     d_Temp, ldt, d_XF_array, ldxf));
};

hipsolverStatus_t hipsolverRfBatchZeroPivot(hipsolverRfHandle_t handle,
                                            /* output in host memory */
                                            int* position)
{
    return ((hipsolverStatus_t)rocsolverRfBatchZeroPivot((rocsolverRfHandle_t)handle, position));
};

#ifdef __cplusplus
};
#endif
