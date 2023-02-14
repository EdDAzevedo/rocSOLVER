
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
#include "hip_check.h"
#include "hipsparse_check.h"

#include "rocsolver_refactor.h"

/*
--------------------------------------------------------------------------
This routine performs the appropriate analysis of parallelism available in
the LU re-factorization depending upon the algorithm chosen by the user.

   A = L * U

It is assumed that a prior call to rocsolverRfSetupHost() or
rocsolverRfSetupDevice() was done in order to create internal data
structures needed for the analysis.

This routine needs to be called only once for a single linear system.
--------------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfBatchAnalyze(rocsolverRfHandle_t handle)
{
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;

    try
    {
        int const ibatch = 0;
        int const nnzLU = handle->nnzLU;
        int const n = handle->n;
        int const batch_count = handle->batch_count;

        int* const csrRowPtrLU = handle->csrRowPtrLU.data().get();
        int* const csrColIndLU = handle->csrColIndLU.data().get();
        double* csrValLU = handle->csrValLU_array.data().get();
        double* const csrValLU_array = handle->csrValLU_array.data().get();

        auto const infoL = handle->infoL.data();
        auto const infoU = handle->infoU.data();
        auto const infoLU = infoLU_array[ibatch].data();

        hipsparseHandle_t hipsparse_handle = handle->hipsparse_handle.data();
        void* const buffer = handle->buffer.data().get();

        hipsparseMatDescr_t const descrL = handle->descrL.data();
        hipsparseMatDescr_t const descrU = handle->descrU.data();
        hipsparseMatDescr_t const descrLU = handle->descrLU.data();
        {
            bool const isok = (n >= 0) && (nnzLU >= 0) && (batch_count >= 0) && (csrRowPtrLU != 0)
                && (csrColIndLU != 0) && (csrValLU_array != 0) && (descrL != 0) && (descrU != 0)
                && (descrLU != 0);

            if(!isok)
            {
                return (ROCSOLVER_STATUS_INTERNAL_ERROR);
            };
        };

        hipsparseSolvePolicy_t const policy = (handle->solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG1)
            ? HIPSPARSE_SOLVE_POLICY_USE_LEVEL
            : HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

        hipsparseOperation_t transL = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        hipsparseOperation_t transU = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        // ----------------
        // perform analysis
        // ----------------

        THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrsv2_analysis(hipsparse_handle, transL, n, nnzLU,
                                                           descrL, csrValLU, csrRowPtrLU,
                                                           csrColIndLU, infoL, policy, buffer));

        THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrsv2_analysis(hipsparse_handle, transU, n, nnzLU,
                                                           descrU, csrValLU, csrRowPtrLU,
                                                           csrColIndLU, infoU, policy, buffer));

        THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrilu02_analysis(hipsparse_handle, n, nnzLU, descrLU,
                                                             csrValLU, csrRowPtrLU, csrColIndLU,
                                                             infoLU, policy, buffer));
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
    return (istat_return);
};
};
