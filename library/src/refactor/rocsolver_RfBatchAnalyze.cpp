
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
    if(handle == 0)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    if(handle->hipsparse_handle == 0)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    int const nnz_LU = handle->nnz_LU;
    int const n = handle->n;
    int const batch_count = handle->batch_count;

    int* csrRowPtrLU = handle->csrRowPtrLU;
    int* csrColIndLU = handle->csrColIndLU;
    double** csrValLU_array = handle->csrValLU_array;

    hipsparseMatDescr_t const descrL = handle->descrL;
    hipsparseMatDescr_t const descrU = handle->descrU;
    hipsparseMatDescr_t const descrLU = handle->descrLU;
    {
        bool const isok = (n >= 0) && (nnz_LU >= 0) && (batch_count >= 0) && (csrRowPtrLU != 0)
            && (csrColIndLU != 0) && (csrValLU_array != 0) && (descrL != 0) && (descrU != 0)
            && (descrLU != 0);

        if(!isok)
        {
            return (ROCSOLVER_STATUS_INTERNAL_ERROR);
        };
    };

    for(int ibatch = 0; ibatch < batch_count; ibatch++)
    {
        double* const csrValLU = csrValLU_array[ibatch];
        if(csrValLU == 0)
        {
            return (ROCSOLVER_STATUS_INTERNAL_ERROR);
        };
    };

    double* const csrValLU = csrValLU_array[0];

    hipsparseSolvePolicy_t const policy = (handle->solve_alg == ROCSOLVERRF_TRIANGULAR_SOLVE_ALG1)
        ? HIPSPARSE_SOLVE_POLICY_USE_LEVEL
        : HIPSPARSE_SOLVE_POLICY_NO_LEVEL;

    hipsparseOperation_t transL = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transU = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    // ---------------------------
    // create infoL, infoU, infoLU
    // ---------------------------
    {
        if(handle->infoL != 0)
        {
            HIPSPARSE_CHECK(hipsparseDestroyCsrsv2Info(handle->infoL),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
            handle->infoL = 0;
        };
        HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&(handle->infoL)), ROCSOLVER_STATUS_INTERNAL_ERROR);

        if(handle->infoU != 0)
        {
            HIPSPARSE_CHECK(hipsparseDestroyCsrsv2Info(handle->infoU),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
            handle->infoU = 0;
        };
        HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&(handle->infoU)), ROCSOLVER_STATUS_INTERNAL_ERROR);

        if(handle->infoLU_array != 0)
        {
            for(int ibatch = 0; ibatch < handle->batch_count; ibatch++)
            {
                if(handle->infoLU_array[ibatch] != 0)
                {
                    HIPSPARSE_CHECK(hipsparseDestroyCsrilu02Info(handle->infoLU_array[ibatch]),
                                    ROCSOLVER_STATUS_INTERNAL_ERROR);
                };
                handle->infoLU_array[ibatch] = 0;
            };

            HIP_CHECK(hipFree(handle->infoLU_array), ROCSOLVER_STATUS_ALLOC_FAILED);
            handle->infoLU_array = 0;
        };

        {
            size_t nbytes = handle->batch_count * sizeof(csrilu02Info_t);
            HIP_CHECK(hipMalloc((void**)&(handle->infoLU_array), nbytes),
                      ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        for(int ibatch = 0; ibatch < handle->batch_count; ibatch++)
        {
            HIPSPARSE_CHECK(hipsparseCreateCsrilu02Info(&(handle->infoLU_array[ibatch])),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
        };
    };

    size_t bufferSize = 1;
    int stmp = 0;

    {
        // ---------------------------------------------
        // check buffer size for triangular solve with L
        // ---------------------------------------------

        stmp = 0;
        HIPSPARSE_CHECK(hipsparseDcsrsv2_bufferSize(handle->hipsparse_handle, transL, n, nnz_LU,
                                                    handle->descrL, csrValLU, csrRowPtrLU,
                                                    csrColIndLU, handle->infoL, &stmp),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);
        if(stmp > bufferSize)
        {
            bufferSize = stmp;
        };
    };

    {
        // ---------------------------------------------
        // check buffer size for triangular solve with U
        // ---------------------------------------------
        stmp = 0;
        HIPSPARSE_CHECK(hipsparseDcsrsv2_bufferSize(handle->hipsparse_handle, transU, n, nnz_LU,
                                                    handle->descrU, csrValLU, csrRowPtrLU,
                                                    csrColIndLU, handle->infoU, &stmp),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        if(stmp > bufferSize)
        {
            bufferSize = stmp;
        };
    };

    {
        // -------------------------
        // check buffer size for ILU
        // -------------------------
        int const ibatch = 0;
        HIPSPARSE_CHECK(hipsparseDcsrilu02_bufferSize(
                            handle->hipsparse_handle, n, nnz_LU, handle->descrLU, csrValLU,
                            csrRowPtrLU, csrColIndLU, handle->infoLU_array[ibatch], &stmp),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);
        if(stmp > bufferSize)
        {
            bufferSize = stmp;
        };
    };

    {
        // ------------------------
        // reuse buffer if possible
        // ------------------------
        if(bufferSize > handle->buffer_size)
        {
            HIP_CHECK(hipFree(handle->buffer), ROCSOLVER_STATUS_INTERNAL_ERROR);
            handle->buffer_size = bufferSize;
            HIP_CHECK(hipMalloc(&(handle->buffer), bufferSize), ROCSOLVER_STATUS_ALLOC_FAILED);

            if(handle->buffer == 0)
            {
                return (ROCSOLVER_STATUS_ALLOC_FAILED);
            };
        };
    };

    {
        // ----------------
        // perform analysis
        // ----------------

        HIPSPARSE_CHECK(hipsparseDcsrsv2_analysis(handle->hipsparse_handle, transL, n, nnz_LU,
                                                  handle->descrL, csrValLU, csrRowPtrLU, csrColIndLU,
                                                  handle->infoL, policy, handle->buffer),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        HIPSPARSE_CHECK(hipsparseDcsrsv2_analysis(handle->hipsparse_handle, transU, n, nnz_LU,
                                                  handle->descrU, csrValLU, csrRowPtrLU, csrColIndLU,
                                                  handle->infoU, policy, handle->buffer),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        {
            int const ibatch = 0;
            HIPSPARSE_CHECK(hipsparseDcsrilu02_analysis(handle->hipsparse_handle, n, nnz_LU,
                                                        handle->descrLU, csrValLU, csrRowPtrLU,
                                                        csrColIndLU, handle->infoLU_array[ibatch],
                                                        policy, handle->buffer),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
        };
    };

    return (ROCSOLVER_STATUS_SUCCESS);
};
}
