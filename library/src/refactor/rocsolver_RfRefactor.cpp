
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
for refactorization and solving routines, rocsolverRfRefactor() and
rocsolverRfSolve().  The wrong combination generates the error code
ROCSOLVER_STATUS_INVALID_VALUE.
---------------------------------------------
 */

extern "C" {

rocsolverStatus_t rocsolverRfRefactor(rocsolverRfHandle_t handle)
{
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    if(handle->hipsparse_handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    int const ibatch = 0;
    csrilu02Info_t info = handle->infoLU_array[ibatch];

    if(info == 0)
    {
        HIPSPARSE_CHECK(hipsparseCreateCsrilu02Info(&info), ROCSOLVER_STATUS_EXECUTION_FAILED);
    };

    int* csrSortedRowPtrA = handle->csrRowPtrLU;
    int* csrSortedColIndA = handle->csrColIndLU;
    double* csrSortedValA = handle->csrValLU_array[ibatch];

    int const n = handle->n;
    int const nnz = handle->nnz_LU;
    hipsparseMatDescr_t const descrA = handle->descrLU;

    int BufferSizeInBytes_int = 0;
    HIPSPARSE_CHECK(hipsparseDcsrilu02_bufferSize(handle->hipsparse_handle, n, nnz, descrA,
                                                  csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                                                  info, &BufferSizeInBytes_int),
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    double* pBuffer = nullptr;
    {
        size_t const BufferSizeInBytes = BufferSizeInBytes_int;
        HIP_CHECK(hipMalloc((void**)&pBuffer, BufferSizeInBytes), ROCSOLVER_STATUS_ALLOC_FAILED);
    };

    /*
	 ---------------------------------------------------------------------------------
	 perform analysis

	 note policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL or HIPSPARSE_SOLVE_POLICY_NO_LEVEL
	 ---------------------------------------------------------------------------------
	 */
    hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

    HIPSPARSE_CHECK(hipsparseDcsrilu02_analysis(handle->hipsparse_handle, n, nnz, descrA,
                                                csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                                                info, policy, pBuffer),
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    {
        /* 
          ----------------------------------------------------
          numerical boost is disabled by default in cusolverRF
          ----------------------------------------------------
         */
        double effective_zero = handle->effective_zero;
        double boost_val = handle->boost_val;
        int enable_boost = (boost_val > 0.0) ? 1 : 0;

        HIPSPARSE_CHECK(hipsparseDcsrilu02_numericBoost(handle->hipsparse_handle, info,
                                                        enable_boost, &effective_zero, &boost_val),
                        ROCSOLVER_STATUS_EXECUTION_FAILED);
    };

    /*
	 ---------------------
	 perform factorization
	 ---------------------
	 */
    HIPSPARSE_CHECK(hipsparseDcsrilu02(handle->hipsparse_handle, n, nnz, descrA, csrSortedValA,
                                       csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer),
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    HIP_CHECK(hipFree(pBuffer), ROCSOLVER_STATUS_ALLOC_FAILED);
    /*
	 --------------------
	 check for zero pivot
	 --------------------
         */
    int pivot = -(n + 1);

    HIPSPARSE_CHECK(hipsparseXcsrilu02_zeroPivot(handle, info, &pivot),
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    handle->infoLU_array[ibatch] = info;

    {
        bool isok = (pivot == -1);
        if(!isok)
        {
            return (ROCSOLVER_STATUS_ZERO_PIVOT);
        };
    };

    return (ROCSOLVER_STATUS_SUCCESS);
};
};
