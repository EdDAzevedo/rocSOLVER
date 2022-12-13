
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
#ifndef ROCSOLVER_REFACTOR_H
#define ROCSOLVER_REFACTOR_H

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include "hipsparse/hipsparse.h"

#include "rocsolver_status.h"


typedef enum
{
    ROCSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0, //default
    ROCSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1
} rocsolverRfResetValuesFastMode_t;

/* ROCSOLVERRF matrix format */
typedef enum
{
    ROCSOLVERRF_MATRIX_FORMAT_CSR = 0, //default
    ROCSOLVERRF_MATRIX_FORMAT_CSC = 1
} rocsolverRfMatrixFormat_t;

/* ROCSOLVERRF unit diagonal */
typedef enum
{
    ROCSOLVERRF_UNIT_DIAGONAL_STORED_L = 0, //default
    ROCSOLVERRF_UNIT_DIAGONAL_STORED_U = 1,
    ROCSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2,
    ROCSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3
} rocsolverRfUnitDiagonal_t;

/* ROCSOLVERRF factorization algorithm */
typedef enum
{
    ROCSOLVERRF_FACTORIZATION_ALG0 = 0, // default
    ROCSOLVERRF_FACTORIZATION_ALG1 = 1,
    ROCSOLVERRF_FACTORIZATION_ALG2 = 2,
} rocsolverRfFactorization_t;

/* ROCSOLVERRF triangular solve algorithm */
typedef enum
{
    ROCSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1, // default
    ROCSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2,
    ROCSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3
} rocsolverRfTriangularSolve_t;

/* ROCSOLVERRF numeric boost report */
typedef enum
{
    ROCSOLVERRF_NUMERIC_BOOST_NOT_USED = 0, //default
    ROCSOLVERRF_NUMERIC_BOOST_USED = 1
} rocsolverRfNumericBoostReport_t;

/* structure holding ROCSOLVERRF library common */
struct rocsolverRfCommon
{
    rocsolverRfResetValuesFastMode_t fast_mode;
    rocsolverRfMatrixFormat_t matrix_format;
    rocsolverRfTriangularSolve_t triangular_solve;
    rocsolverRfNumericBoostReport_t numeric_boost;

    hipsparseHandle_t hipsparse_handle;

    hipsparseMatDescr_t descrL;
    hipsparseMatDescr_t descrU;
    hipsparseMatDescr_t descrLU;

    int* P_new2old;
    int* Q_new2old;
    int* Q_old2new;

    int n;
    int nnz_LU;
    int* csrRowPtrLU;
    int* csrColIndLU;
    double* csrValLU;

    double effective_zero;
    double boost_val;
};
typedef struct rocsolverRfCommon* rocsolverRfHandle_t;

#ifdef __cplusplus
extern "C" {
#endif

/* ROCSOLVERRF create (allocate memory) and destroy (free memory) in the handle */
rocsolverStatus_t rocsolverRfCreate(rocsolverRfHandle_t* handle);
rocsolverStatus_t rocsolverRfDestroy(rocsolverRfHandle_t handle);

/* ROCSOLVERRF set and get input format */
rocsolverStatus_t rocsolverRfGetMatrixFormat(rocsolverRfHandle_t handle,
                                             rocsolverRfMatrixFormat_t* format,
                                             rocsolverRfUnitDiagonal_t* diag);

rocsolverStatus_t rocsolverRfSetMatrixFormat(rocsolverRfHandle_t handle,
                                             rocsolverRfMatrixFormat_t format,
                                             rocsolverRfUnitDiagonal_t diag);

/* ROCSOLVERRF set and get numeric properties */
rocsolverStatus_t
    rocsolverRfSetNumericProperties(rocsolverRfHandle_t handle, double zero, double boost);

rocsolverStatus_t
    rocsolverRfGetNumericProperties(rocsolverRfHandle_t handle, double* zero, double* boost);

rocsolverStatus_t rocsolverRfGetNumericBoostReport(rocsolverRfHandle_t handle,
                                                   rocsolverRfNumericBoostReport_t* report);

/* ROCSOLVERRF choose the triangular solve algorithm */
rocsolverStatus_t rocsolverRfSetAlgs(rocsolverRfHandle_t handle,
                                     rocsolverRfFactorization_t factAlg,
                                     rocsolverRfTriangularSolve_t solveAlg);

rocsolverStatus_t rocsolverRfGetAlgs(rocsolverRfHandle_t handle,
                                     rocsolverRfFactorization_t* factAlg,
                                     rocsolverRfTriangularSolve_t* solveAlg);

/* ROCSOLVERRF set and get fast mode */
rocsolverStatus_t rocsolverRfGetResetValuesFastMode(rocsolverRfHandle_t handle,
                                                    rocsolverRfResetValuesFastMode_t* fastMode);

rocsolverStatus_t rocsolverRfSetResetValuesFastMode(rocsolverRfHandle_t handle,
                                                    rocsolverRfResetValuesFastMode_t fastMode);

/*** Non-Batched Routines ***/
/* ROCSOLVERRF setup of internal structures from host or device memory */
rocsolverStatus_t rocsolverRfSetupHost(/* Input (in the host memory) */
                                       int n,
                                       int nnzA,
                                       int* h_csrRowPtrA,
                                       int* h_csrColIndA,
                                       double* h_csrValA,
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
                                       /* Output */
                                       rocsolverRfHandle_t handle);

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

/* ROCSOLVERRF update the matrix values (assuming the reordering, pivoting 
   and consequently the sparsity pattern of L and U did not change),
   and zero out the remaining values. */
rocsolverStatus_t rocsolverRfResetValues(/* Input (in the device memory) */
                                         int n,
                                         int nnzA,
                                         int* csrRowPtrA,
                                         int* csrColIndA,
                                         double* csrValA,
                                         int* P,
                                         int* Q,
                                         /* Output */
                                         rocsolverRfHandle_t handle);

/* ROCSOLVERRF analysis (for parallelism) */
rocsolverStatus_t rocsolverRfAnalyze(rocsolverRfHandle_t handle);

/* ROCSOLVERRF re-factorization (for parallelism) */
rocsolverStatus_t rocsolverRfRefactor(rocsolverRfHandle_t handle);

/* ROCSOLVERRF extraction: Get L & U packed into a single matrix M */
rocsolverStatus_t rocsolverRfAccessBundledFactorsDevice(/* Input */
                                                        rocsolverRfHandle_t handle,
                                                        /* Output (in the host memory) */
                                                        int* nnzM,
                                                        /* Output (in the device memory) */
                                                        int** Mp,
                                                        int** Mi,
                                                        double** Mx);

rocsolverStatus_t rocsolverRfExtractBundledFactorsHost(/* Input */
                                                       rocsolverRfHandle_t handle,
                                                       /* Output (in the host memory) */
                                                       int* h_nnzM,
                                                       int** h_Mp,
                                                       int** h_Mi,
                                                       double** h_Mx);

/* ROCSOLVERRF extraction: Get L & U individually */
rocsolverStatus_t rocsolverRfExtractSplitFactorsHost(/* Input */
                                                     rocsolverRfHandle_t handle,
                                                     /* Output (in the host memory) */
                                                     int* h_nnzL,
                                                     int** h_csrRowPtrL,
                                                     int** h_csrColIndL,
                                                     double** h_csrValL,
                                                     int* h_nnzU,
                                                     int** h_csrRowPtrU,
                                                     int** h_csrColIndU,
                                                     double** h_csrValU);

/* ROCSOLVERRF (forward and backward triangular) solves */
rocsolverStatus_t rocsolverRfSolve(/* Input (in the device memory) */
                                   rocsolverRfHandle_t handle,
                                   int* P,
                                   int* Q,
                                   int nrhs, //only nrhs=1 is supported
                                   double* Temp, //of size ldt*nrhs (ldt>=n)
                                   int ldt,
                                   /* Input/Output (in the device memory) */
                                   double* XF,
                                   /* Input */
                                   int ldxf);

/*** Batched Routines ***/
/* ROCSOLVERRF-batch setup of internal structures from host */
rocsolverStatus_t rocsolverRfBatchSetupHost(/* Input (in the host memory)*/
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
                                            /* Output (in the device memory) */
                                            rocsolverRfHandle_t handle);

/* ROCSOLVERRF-batch update the matrix values (assuming the reordering, pivoting 
   and consequently the sparsity pattern of L and U did not change),
   and zero out the remaining values. */
rocsolverStatus_t rocsolverRfBatchResetValues(/* Input (in the device memory) */
                                              int batchSize,
                                              int n,
                                              int nnzA,
                                              int* csrRowPtrA,
                                              int* csrColIndA,
                                              double* csrValA_array[],
                                              int* P,
                                              int* Q,
                                              /* Output */
                                              rocsolverRfHandle_t handle);

/* ROCSOLVERRF-batch analysis (for parallelism) */
rocsolverStatus_t rocsolverRfBatchAnalyze(rocsolverRfHandle_t handle);

/* ROCSOLVERRF-batch re-factorization (for parallelism) */
rocsolverStatus_t rocsolverRfBatchRefactor(rocsolverRfHandle_t handle);

/* ROCSOLVERRF-batch (forward and backward triangular) solves */
rocsolverStatus_t rocsolverRfBatchSolve(/* Input (in the device memory) */
                                        rocsolverRfHandle_t handle,
                                        int* P,
                                        int* Q,
                                        int nrhs, //only nrhs=1 is supported
                                        double* Temp, //of size 2*batchSize*(n*nrhs)
                                        int ldt, //only ldt=n is supported
                                        /* Input/Output (in the device memory) */
                                        double* XF_array[],
                                        /* Input */
                                        int ldxf);

/* ROCSOLVERRF-batch obtain the position of zero pivot */
rocsolverStatus_t rocsolverRfBatchZeroPivot(/* Input */
                                            rocsolverRfHandle_t handle,
                                            /* Output (in the host memory) */
                                            int* position);

#ifdef __cplusplus
}
#endif

#endif /* ROCSOLVER_REFACTOR_H */
