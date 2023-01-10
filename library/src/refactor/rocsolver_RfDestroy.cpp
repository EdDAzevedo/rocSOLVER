/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
-----------------------------------------------------------------------
This routine shuts down the rocSolverRF library.  It releases acquired
resources and must be called after all the rocsolverRF library routines.
-----------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfDestroy(rocsolverRfHandle_t handle)
{
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    if(handle->hipsparse_handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    // deallocate Mat Descr
    if(handle->descrL != nullptr)
    {
        HIPSPARSE_CHECK(hipsparseDestroyMatDescr(handle->descrL), ROCSOLVER_STATUS_INTERNAL_ERROR);
        handle->descrL = nullptr;
    };

    if(handle->descrU != nullptr)
    {
        HIPSPARSE_CHECK(hipsparseDestroyMatDescr(handle->descrU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        handle->descrU = nullptr;
    };

    if(handle->descrLU != nullptr)
    {
        HIPSPARSE_CHECK(hipsparseDestroyMatDescr(handle->descrLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        handle->descrLU = nullptr;
    };

    // -----------------------
    // deallocate permutations
    // -----------------------
    if(handle->P_new2old != nullptr)
    {
        HIP_CHECK(hipFree(handle->P_new2old), ROCSOLVER_STATUS_INTERNAL_ERROR);
        handle->P_new2old = nullptr;
    };

    if(handle->Q_new2old != nullptr)
    {
        HIP_CHECK(hipFree(handle->Q_new2old), ROCSOLVER_STATUS_INTERNAL_ERROR);
        handle->Q_new2old = nullptr;
    };

    if(handle->Q_old2new != nullptr)
    {
        HIP_CHECK(hipFree(handle->Q_old2new), ROCSOLVER_STATUS_INTERNAL_ERROR);
        handle->Q_old2new = nullptr;
    };

    // ---------------------------
    // deallocate sparse matrix LU
    // ---------------------------
    if(handle->csrRowPtrLU != nullptr)
    {
        HIP_CHECK(hipFree(handle->csrRowPtrLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        handle->csrRowPtrLU = nullptr;
    };

    if(handle->csrColIndLU != nullptr)
    {
        HIP_CHECK(hipFree(handle->csrColIndLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        handle->csrColIndLU = nullptr;
    };

    int const batch_count = handle->batch_count;
    handle->batch_count = 0;

    {
        if(handle->csrValLU_array != nullptr)
        {
            for(int ibatch = 0; ibatch < batch_count; ibatch++)
            {
                if(handle->csrValLU_array[ibatch] != nullptr)
                {
                    HIP_CHECK(hipFree(handle->csrValLU_array[ibatch]),
                              ROCSOLVER_STATUS_INTERNAL_ERROR);
                    handle->csrValLU_array[ibatch] = nullptr;
                };
            };

            HIP_CHECK(hipFree(handle->csrValLU_array), ROCSOLVER_STATUS_INTERNAL_ERROR);
            handle->csrValLU_array = nullptr;
        };
    };

    // ---------------
    // deallocate info
    // ---------------

    {
        if(handle->infoL != 0)
        {
            HIPSPARSE_CHECK(hipsparseDestroyCsrsv2Info(handle->infoL),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
            handle->infoL = 0;
        };

        if(handle->infoU != 0)
        {
            HIPSPARSE_CHECK(hipsparseDestroyCsrsv2Info(handle->infoU),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
            handle->infoU = 0;
        };
    };

    {
        csrilu02Info_t* infoLU_array = handle->infoLU_array;
        if(handle->infoLU_array != nullptr)
        {
            for(int ibatch = 0; ibatch < batch_count; ibatch++)
            {
                csrilu02Info_t infoLU = infoLU_array[ibatch];
                HIPSPARSE_CHECK(hipsparseDestroyCsrilu02Info(infoLU),
                                ROCSOLVER_STATUS_INTERNAL_ERROR);
                infoLU_array[ibatch] = 0;
            };

            HIP_CHECK(hipFree(handle->infoLU_array), ROCSOLVER_STATUS_INTERNAL_ERROR);
            handle->infoLU_array = nullptr;
        };
    };

    if(handle->buffer != nullptr)
    {
        HIP_CHECK(hipFree(handle->buffer), ROCSOLVER_STATUS_INTERNAL_ERROR);
        handle->buffer = nullptr;
    };

    handle->n = 0;
    handle->nnz_LU = 0;

    // -------------------------
    // free the hipsparse handle
    // -------------------------
    {
        HIPSPARSE_CHECK(hipsparseDestroy(handle->hipsparse_handle), ROCSOLVER_STATUS_INTERNAL_ERROR);
    };

    // -------------------------------
    // finally free the handle on Host
    // -------------------------------
    HIP_CHECK(hipHostFree(handle), ROCSOLVER_STATUS_INTERNAL_ERROR);
    handle = nullptr;
    return (ROCSOLVER_STATUS_SUCCESS);
};
};
