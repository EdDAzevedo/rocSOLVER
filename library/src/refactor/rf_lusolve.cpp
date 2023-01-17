
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
#include "rf_lusolve.h"

rocsolverStatus_t rf_lusolve(hipsparseHandle_t handle,
                             int const n,
                             int const nnz,
                             int* const d_LUp,
                             int* const d_LUi,
                             double* const d_LUx,
                             double* const d_b)
{
    int const m = n;
    {
        bool const isok_arg
            = (d_LUp != nullptr) && (d_LUi != nullptr) && (d_LUx != nullptr) && (d_b != nullptr);

        if(!isok_arg)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    {
        bool const isok_pointer = (is_device_pointer(d_LUp)) && (is_device_pointer(d_LUi))
            && (is_device_pointer(d_LUx)) && (is_device_pointer(d_b));

        if(!isok_pointer)
        {
            return (ROCSOLVER_STATUS_INTERNAL_ERROR);
        };
    };

    /*
   ----------------------------------------------------
   Create L factor descriptor and triangular solve info
   ----------------------------------------------------
   */
    hipsparseMatDescr_t descrL;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrL), ROCSOLVER_STATUS_INTERNAL_ERROR);
    HIPSPARSE_CHECK(hipsparseSetMatType(descrL, HIPSPARSE_MATRIX_TYPE_GENERAL),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);
    HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descrL, HIPSPARSE_INDEX_BASE_ZERO),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);
    HIPSPARSE_CHECK(hipsparseSetMatFillMode(descrL, HIPSPARSE_FILL_MODE_LOWER),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);
    HIPSPARSE_CHECK(hipsparseSetMatDiagType(descrL, HIPSPARSE_DIAG_TYPE_UNIT),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);

    csrsv2Info_t infoL;
    HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&infoL), ROCSOLVER_STATUS_INTERNAL_ERROR);

    /*
    ----------------------------------------------------
    Create U factor descriptor and triangular solve info
    ----------------------------------------------------
   */
    hipsparseMatDescr_t descrU;
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrU), ROCSOLVER_STATUS_INTERNAL_ERROR);
    HIPSPARSE_CHECK(hipsparseSetMatType(descrU, HIPSPARSE_MATRIX_TYPE_GENERAL),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);
    HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descrU, HIPSPARSE_INDEX_BASE_ZERO),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);
    HIPSPARSE_CHECK(hipsparseSetMatFillMode(descrU, HIPSPARSE_FILL_MODE_UPPER),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);
    HIPSPARSE_CHECK(hipsparseSetMatDiagType(descrU, HIPSPARSE_DIAG_TYPE_NON_UNIT),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);

    csrsv2Info_t infoU;
    HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&infoU), ROCSOLVER_STATUS_INTERNAL_ERROR);

    /*
   --------------------------------
   Allocate workspace for hipSPARSE
   --------------------------------
   */

    int const lnnz = nnz;
    size_t bufferSize = 1;
    int stmp = 0;

    double* const csrSortedValA = d_LUx;
    int* const csrSortedRowPtrA = d_LUp;
    int* const csrSortedColIndA = d_LUi;

    hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
    hipsparseOperation_t transL = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transU = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    stmp = 0;
    HIPSPARSE_CHECK(hipsparseDcsrsv2_bufferSize(handle, transL, m, lnnz, descrL, csrSortedValA,
                                                csrSortedRowPtrA, csrSortedColIndA, infoL, &stmp),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);

    if(stmp > bufferSize)
    {
        bufferSize = stmp;
    }

    stmp = 0;
    HIPSPARSE_CHECK(hipsparseDcsrsv2_bufferSize(handle, transU, m, lnnz, descrU, csrSortedValA,
                                                csrSortedRowPtrA, csrSortedColIndA, infoU, &stmp),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);

    if(stmp > bufferSize)
    {
        bufferSize = stmp;
    }

    void* buffer = nullptr;
    HIP_CHECK(hipMalloc(&buffer, bufferSize), ROCSOLVER_STATUS_ALLOC_FAILED);

    double* d_x = d_b;
    double* d_y = nullptr;
    HIP_CHECK(hipMalloc((void**)&d_y, sizeof(double) * m), ROCSOLVER_STATUS_ALLOC_FAILED);
    if(d_y == nullptr)
    {
        return (ROCSOLVER_STATUS_ALLOC_FAILED);
    };

    /*
   -------------------------------------------
   If A = LU
   Solve A x = (LU) x b as
   (1)   solve L y = b,   L unit diagonal
   (2)   solve U x = y,   U non-unit diagonal

   -------------------------------------------
   */

    HIPSPARSE_CHECK(hipsparseDcsrsv2_analysis(handle, transL, m, lnnz, descrL, csrSortedValA,
                                              csrSortedRowPtrA, csrSortedColIndA, infoL, policy,
                                              buffer),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);

    HIPSPARSE_CHECK(hipsparseDcsrsv2_analysis(handle, transU, m, lnnz, descrU, csrSortedValA,
                                              csrSortedRowPtrA, csrSortedColIndA, infoU, policy,
                                              buffer),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);

    /*
     ----------------------
     step (1) solve L y = b
     ----------------------
     */

    double alpha = 1.0;

    HIPSPARSE_CHECK(hipsparseDcsrsv2_solve(handle, transL, m, lnnz, &alpha, descrL, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, infoL, d_b, d_y,
                                           policy, buffer),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);

    /*
          ----------------------
          step (2) solve U x = y
          ----------------------
         */

    HIPSPARSE_CHECK(hipsparseDcsrsv2_solve(handle, transU, m, lnnz, &alpha, descrU, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, infoU, d_y, d_x,
                                           policy, buffer),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);

    /*
   --------
   clean up
   --------
   */

    HIPSPARSE_CHECK(hipsparseDestroyCsrsv2Info(infoU), ROCSOLVER_STATUS_INTERNAL_ERROR);

    HIPSPARSE_CHECK(hipsparseDestroyCsrsv2Info(infoL), ROCSOLVER_STATUS_INTERNAL_ERROR);

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrL), ROCSOLVER_STATUS_INTERNAL_ERROR);

    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrU), ROCSOLVER_STATUS_INTERNAL_ERROR);

    HIP_CHECK(hipFree((void*)buffer), ROCSOLVER_STATUS_ALLOC_FAILED);
    HIP_CHECK(hipFree((void*)d_y), ROCSOLVER_STATUS_ALLOC_FAILED);

    return (ROCSOLVER_STATUS_SUCCESS);
}
