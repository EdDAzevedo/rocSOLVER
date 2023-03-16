
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
    int constexpr idebug = 0;

    {
        bool isok = (handle != nullptr);
        if(!isok)
        {
            return (ROCSOLVER_STATUS_NOT_INITIALIZED);
        };
    }

    Iint const m = n;
    {
        bool const isok_scalar = (n >= 0) && (nnz >= 0);
        bool const isok_arg = (d_LUp != nullptr) && (d_LUi != nullptr) && (d_LUx != nullptr)
            && (d_b != nullptr) && (d_Temp != nullptr);

        bool const isok_all = isok_arg && isok_scalar;
        if(!isok_all)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    if(idebug >= 1)
    {
        printf("%s:%d\n", __FILE__, __LINE__);
        fflush(stdout);
    };

    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;
    try
    {
        hipsparseHandle_t hipsparse_handle = handle->hipsparse_handle.data();

        hipsparseMatDescr_t descrL = handle->descrL.data();

        csrsv2Info_t infoL = handle->infoL.data();

        hipsparseMatDescr_t descrU = handle->descrU.data();

        csrsv2Info_t infoU = handle->infoU.data();

        {
            bool const isok = (hipsparse_handle != nullptr) && (descrL != nullptr)
                && (infoL != nullptr) && (descrU != nullptr) && (infoU != nullptr);
            RF_ASSERT(isok);
        };

        if(idebug >= 1)
        {
            printf("%s:%d\n", __FILE__, __LINE__);
            fflush(stdout);
        };

        // --------------------------------
        // Allocate workspace for hipSPARSE
        // --------------------------------

        Ilong const lnnz = nnz;
        int stmp_L = 0;
        int stmp_U = 0;

        T* const csrSortedValA = d_LUx;
        Ilong* const csrSortedRowPtrA = d_LUp;
        Iint* const csrSortedColIndA = d_LUi;

        hipsparseSolvePolicy_t policy = (handle->solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG1)
            ? HIPSPARSE_SOLVE_POLICY_USE_LEVEL
            : HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

        hipsparseOperation_t transL = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        hipsparseOperation_t transU = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        if(idebug >= 1)
        {
            printf("%s:%d\n", __FILE__, __LINE__);
            fflush(stdout);
        };

        THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrsv2_bufferSize(hipsparse_handle, transL, m, lnnz,
                                                             descrL, csrSortedValA, csrSortedRowPtrA,
                                                             csrSortedColIndA, infoL, &stmp_L));

        if(idebug >= 1)
        {
            printf("%s:%d\n", __FILE__, __LINE__);
            fflush(stdout);
        };

        THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrsv2_bufferSize(hipsparse_handle, transU, m, lnnz,
                                                             descrU, csrSortedValA, csrSortedRowPtrA,
                                                             csrSortedColIndA, infoU, &stmp_U));
        if(idebug >= 1)
        {
            printf("%s:%d\n", __FILE__, __LINE__);
            fflush(stdout);
        };

        int const bufferSize = std::max(stmp_L, stmp_U);
        bool const isok_size = (handle->buffer.size() >= bufferSize);
        if(!isok_size)
        {
            handle->buffer.resize(bufferSize);
        };

        T* const d_y = d_Temp;

        void* const buffer = handle->buffer.data().get();

        T* const d_x = d_b;

        // -------------------------------------------
        // If A = LU
        // Solve A x = (LU) x b as
        // (1)   solve L y = b,   L unit diagonal
        // (2)   solve U x = y,   U non-unit diagonal
        // -------------------------------------------

        if(idebug >= 1)
        {
            printf("%s:%d\n", __FILE__, __LINE__);
            fflush(stdout);
        };

        THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrsv2_analysis(hipsparse_handle, transL, m, lnnz,
                                                           descrL, csrSortedValA, csrSortedRowPtrA,
                                                           csrSortedColIndA, infoL, policy, buffer));

        if(idebug >= 1)
        {
            printf("%s:%d\n", __FILE__, __LINE__);
            fflush(stdout);
        };

        THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrsv2_analysis(hipsparse_handle, transU, m, lnnz,
                                                           descrU, csrSortedValA, csrSortedRowPtrA,
                                                           csrSortedColIndA, infoU, policy, buffer));

        if(idebug >= 1)
        {
            printf("%s:%d\n", __FILE__, __LINE__);
            fflush(stdout);
        };

        // ----------------------
        // step (1) solve L y = b
        // ----------------------

        T alpha = 1.0;

        THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrsv2_solve(
            hipsparse_handle, transL, m, lnnz, &alpha, descrL, csrSortedValA, csrSortedRowPtrA,
            csrSortedColIndA, infoL, d_b, d_y, policy, buffer));

        if(idebug >= 1)
        {
            printf("%s:%d\n", __FILE__, __LINE__);
            fflush(stdout);
        };
        // ----------------------
        // step (2) solve U x = y
        // ----------------------

        THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrsv2_solve(
            hipsparse_handle, transU, m, lnnz, &alpha, descrU, csrSortedValA, csrSortedRowPtrA,
            csrSortedColIndA, infoU, d_y, d_x, policy, buffer));
    }
    catch(const std::bad_alloc& e)
    {
        istat_return = ROCSOLVER_STATUS_ALLOC_FAILED;
    }
    catch(const std::runtime_error& e)
    {
        istat_return = ROCSOLVER_STATUS_EXECUTION_FAILED;
    }
    catch(...)
    {
        istat_return = ROCSOLVER_STATUS_INTERNAL_ERROR;
    };

    if(idebug >= 1)
    {
        printf("%s:%d, istat_return=%d\n", __FILE__, __LINE__, istat_return);
        fflush(stdout);
    };
    return (istat_return);
}
#endif
