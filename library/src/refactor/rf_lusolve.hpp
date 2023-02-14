
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
#pragma once
#ifndef RF_LUSOLVE_HPP
#define RF_LUSOLVE_HPP

template <typename Iint, typename Ilong, typename T>
rocsolverStatus_t rf_lusolve(rocsolverRfHandle_t handle,
                             Iint const n,
                             Ilong const nnz,
                             Ilong* const d_LUp,
                             Iint* const d_LUi,
                             T* const d_LUx,
                             T* const d_b,
                             T* const d_Temp)
{
    {
        bool isok = (handle != nullptr);
        if(!isok)
        {
            return (ROCSOLVER_STATUS_NOT_INITIALIZED);
        };
    }

    Iint const m = n;
    {
        bool const isok_arg
            = (d_LUp != nullptr) && (d_LUi != nullptr) && (d_LUx != nullptr) && (d_b != nullptr);

        if(!isok_arg)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    hipsparseMatDescr_t descrL = handle->descrL.data();

    csrsv2Info_t infoL = handle->infoL.data();

    hipsparseMatDescr_t descrU = handle->descrU.data();

    csrsv2Info_t infoU = handle->infoU.data();

    // --------------------------------
    // Allocate workspace for hipSPARSE
    // --------------------------------

    Ilong const lnnz = nnz;
    size_t bufferSize = 1;
    int stmp = 0;

    T* const csrSortedValA = d_LUx;
    Ilong* const csrSortedRowPtrA = d_LUp;
    Iint* const csrSortedColIndA = d_LUi;

    hipsparseSolvePolicy_t policy = (handle->solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG1)
        ? HIPSPARSE_SOLVE_POLICY_USE_LEVEL
        : HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

    hipsparseOperation_t transL = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transU = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    stmp = 0;
    HIPSPARSE_CHECK(hipsparseDcsrsv2_bufferSize(handle, transL, m, lnnz, descrL, csrSortedValA,
                                                csrSortedRowPtrA, csrSortedColIndA, infoL, &stmp),
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    if(stmp > bufferSize)
    {
        bufferSize = stmp;
    }

    stmp = 0;
    HIPSPARSE_CHECK(hipsparseDcsrsv2_bufferSize(handle, transU, m, lnnz, descrU, csrSortedValA,
                                                csrSortedRowPtrA, csrSortedColIndA, infoU, &stmp),
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    if(stmp > bufferSize)
    {
        bufferSize = stmp;
    }
    bool const isok_size = (handle->buffer_size >= bufferSize);
    assert(isok_size);

    T* const d_y = d_Temp;

    void* const buffer = handle->buffer.data().get();

    T* const d_x = d_b;

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
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    HIPSPARSE_CHECK(hipsparseDcsrsv2_analysis(handle, transU, m, lnnz, descrU, csrSortedValA,
                                              csrSortedRowPtrA, csrSortedColIndA, infoU, policy,
                                              buffer),
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    /*
     ----------------------
     step (1) solve L y = b
     ----------------------
     */

    T alpha = 1.0;

    HIPSPARSE_CHECK(hipsparseDcsrsv2_solve(handle, transL, m, lnnz, &alpha, descrL, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, infoL, d_b, d_y,
                                           policy, buffer),
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    /*
          ----------------------
          step (2) solve U x = y
          ----------------------
         */

    HIPSPARSE_CHECK(hipsparseDcsrsv2_solve(handle, transU, m, lnnz, &alpha, descrU, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, infoU, d_y, d_x,
                                           policy, buffer),
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    return (ROCSOLVER_STATUS_SUCCESS);
}
#endif
