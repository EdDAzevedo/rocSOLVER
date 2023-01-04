
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

#include "rf_common.hpp"

/*
---------------------------------------------
This routine performs the LU re-factorization
    A = L * U
uses the available parallelism on the GPU.  It is assumed that a prior
call to rocsolverRfAnalyze() was done in order to find the available
parallelism.

This routine may be called multiple times, once for each of the linear
systems:
   A_i  x_i = f_i

There are some constraints to the combination of algorithms used
for refactorization and solving routines, rocsolverRfBatchRefactor() and
rocsolverRfSolve().  The wrong combination generates the error code
ROCSOLVER_STATUS_INVALID_VALUE.
---------------------------------------------
 */

extern "C" {

rocsolverStatus_t rocsolverRfBatchRefactor(rocsolverRfHandle_t handle)
{
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    if(handle->hipsparse_handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    int* csrSortedRowPtrA = handle->csrRowPtrLU;
    int* csrSortedColIndA = handle->csrColIndLU;

    int const batch_count = handle->batch_count;

    int const n = handle->n;
    int const nnz = handle->nnz_LU;
    hipsparseMatDescr_t const descrA = handle->descrLU;

    int BufferSizeInBytes_int = 1;
    for(int ibatch = 0; ibatch < batch_count; ibatch++)
    {
        int isize = 1;
        double* csrSortedValA = handle->csrValLU_array[ibatch];

        HIPSPARSE_CHECK(hipsparseDcsrilu02_bufferSize(handle->hipsparse_handle, n, nnz, descrA,
                                                      csrSortedValA, csrSortedRowPtrA,
                                                      csrSortedColIndA,
                                                      handle->infoLU_array[ibatch], &isize),
                        ROCSOLVER_STATUS_EXECUTION_FAILED);
        BufferSizeInBytes_int = max(isize, BufferSizeInBytes_int);
    };

    double* pBuffer = nullptr;
    {
        size_t const BufferSizeInBytes = BufferSizeInBytes_int;
        HIP_CHECK(hipMalloc((void**)&pBuffer, BufferSizeInBytes), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(pBuffer == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };
    };

    // ---------------------------------------------------------------------------------
    // perform analysis
    //
    // note policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL or HIPSPARSE_SOLVE_POLICY_NO_LEVEL
    // ---------------------------------------------------------------------------------
    hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

    for(int ibatch = 0; ibatch < batch_count; ibatch++)
    {
        double* csrSortedValA = handle->csrValLU_array[ibatch];
        HIPSPARSE_CHECK(hipsparseDcsrilu02_analysis(handle->hipsparse_handle, n, nnz, descrA,
                                                    csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                                                    handle->infoLU_array[ibatch], policy, pBuffer),
                        ROCSOLVER_STATUS_EXECUTION_FAILED);
    };

    //  ----------------------------------------------------
    //  numerical boost is disabled by default in cusolverRF
    //  ----------------------------------------------------
    double effective_zero = handle->effective_zero;
    double boost_val = handle->boost_val;

    // -------------------------------------------------------------
    // set enable_boost to 1 or disable by setting enable_boost to 0
    // -------------------------------------------------------------
    int const enable_boost = (boost_val > 0.0) ? 1 : 0;

    // ---------------------
    // perform factorization
    // ---------------------

    for(int ibatch = 0; ibatch < batch_count; ibatch++)
    {
        double* csrSortedValA = handle->csrValLU_array[ibatch];

        HIPSPARSE_CHECK(hipsparseDcsrilu02_numericBoost(handle->hipsparse_handle,
                                                        handle->infoLU_array[ibatch], enable_boost,
                                                        &effective_zero, &boost_val),
                        ROCSOLVER_STATUS_EXECUTION_FAILED);

        HIPSPARSE_CHECK(hipsparseDcsrilu02(handle->hipsparse_handle, n, nnz, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA,
                                           handle->infoLU_array[ibatch], policy, pBuffer),
                        ROCSOLVER_STATUS_EXECUTION_FAILED);
    };

    // ----------------------------------------------------------------------
    // note, there is no checking for zero pivot
    // the user has to call "RfBatchZeroPivot()" to know which matrix failed
    // the LU factorization
    // ----------------------------------------------------------------------

    HIP_CHECK(hipFree(pBuffer), ROCSOLVER_STATUS_EXECUTION_FAILED);
    pBuffer = nullptr;

    return (ROCSOLVER_STATUS_SUCCESS);
};
};
