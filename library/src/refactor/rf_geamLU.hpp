
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
#ifndef RF_GEAMLU_HPP
#define RF_GEAMLU_HPP

#include "rf_common.hpp"

template <typename Iint, typename Ilong, typename T>
rocsolverStatus_t rf_geamLU(hipsparseHandle_t hipsparse_handle,
                            Iint const nrow,
                            Iint const ncol,

                            Ilong const* const Lp,
                            Iint const* const Li,
                            T const* const Lx,

                            Ilong const* const Up,
                            Iint const* const Ui,
                            T const* const Ux,

                            Ilong* const LUp,
                            Iint* const LUi,
                            T* const LUx

)
{
    //  ----------------
    //  form (L - I) + U
    //  assume storage for LUp, LUi, LUx has been allocated
    // ---------------------------------------------------

    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;

    T const alpha = 1;
    T const beta = 1;

    hipsparsePointerMode_t pointer_mode;
    hipsparseStatus_t const istat_get_pointer_mode
        = hipsparseGetPointerMode(hipsparse_handle, &pointer_mode);

    if(istat_get_pointer_mode != HIPSPARSE_STATUS_SUCCESS)
    {
        istat_return = (istat_return == ROCSOLVER_STATUS_SUCCESS) ? ROCSOLVER_STATUS_EXECUTION_FAILED
                                                                  : istat_return;
    };

    hipsparseStatus_t const istat_set_pointer_mode
        = hipsparseSetPointerMode(hipsparse_handle, HIPSPARSE_POINTER_MODE_HOST);

    if(istat_get_pointer_mode != HIPSPARSE_STATUS_SUCCESS)
    {
        istat_return = (istat_return == ROCSOLVER_STATUS_SUCCESS) ? ROCSOLVER_STATUS_EXECUTION_FAILED
                                                                  : istat_return;
    };

    // ------------
    // setup buffer
    // ------------
    void* pBuffer = nullptr;
    size_t bufferSizeInBytes = sizeof(T);

    // ------------------------------
    // hipsparseXcsrgeam2() computes
    // C = alpha * A + beta * B
    // ------------------------------

    hipsparseMatDescr_t descrL;
    hipsparseMatDescr_t descrU;

    {
        hipsparseStatus_t const istat_CreateMatDescr_L = hipsparseCreateMatDescr(&descrL);

        hipsparseStatus_t const istat_SetMatType_L
            = hipsparseSetMatType(descrL, HIPSPARSE_MATRIX_TYPE_GENERAL);

        hipsparseStatus_t const istat_SetMatIndexBase_L
            = hipsparseSetMatIndexBase(descrL, HIPSPARSE_INDEX_BASE_ZERO);

        hipsparseStatus_t const istat_SetMatFillMode_L
            = hipsparseSetMatFillMode(descrL, HIPSPARSE_FILL_MODE_LOWER);

        hipsparseStatus_t const istat_SetMatDiagType_L
            = hipsparseSetMatDiagType(descrL, HIPSPARSE_DIAG_TYPE_UNIT);

        bool const isok_descrL = (istat_CreateMatDescr_L == HIPSPARSE_STATUS_SUCCESS)
            && (istat_SetMatType_L == HIPSPARSE_STATUS_SUCCESS)
            && (istat_SetMatIndexBase_L == HIPSPARSE_STATUS_SUCCESS)
            && (istat_SetMatFillMode_L == HIPSPARSE_STATUS_SUCCESS)
            && (istat_SetMatDiagType_L == HIPSPARSE_STATUS_SUCCESS);

        hipsparseStatus_t istat_CreateMatDescr_U = hipsparseCreateMatDescr(&descrU);

        hipsparseStatus_t istat_SetMatType_U
            = hipsparseSetMatType(descrU, HIPSPARSE_MATRIX_TYPE_GENERAL);

        hipsparseStatus_t istat_SetMatIndexBase_U
            = hipsparseSetMatIndexBase(descrU, HIPSPARSE_INDEX_BASE_ZERO);

        hipsparseStatus_t istat_SetMatFillMode_U
            = hipsparseSetMatFillMode(descrU, HIPSPARSE_FILL_MODE_UPPER);

        hipsparseStatus_t istat_SetMatDiagType_U
            = hipsparseSetMatDiagType(descrU, HIPSPARSE_DIAG_TYPE_NON_UNIT);

        bool const isok_descrU = (istat_CreateMatDescr_U == HIPSPARSE_STATUS_SUCCESS)
            && (istat_SetMatType_U == HIPSPARSE_STATUS_SUCCESS)
            && (istat_SetMatIndexBase_U == HIPSPARSE_STATUS_SUCCESS)
            && (istat_SetMatFillMode_U == HIPSPARSE_STATUS_SUCCESS)
            && (istat_SetMatDiagType_U == HIPSPARSE_STATUS_SUCCESS);

        bool const isok_descr_all = isok_descrU && isok_descrL;
        if(!isok_descr_all)
        {
            istat_return = ROCSOLVER_STATUS_EXECUTION_FAILED;
        };
    };

    hipsparseStatus_t istat_geam2_bufferSizeExt = hipsparseDcsrgeam2_bufferSizeExt(
        hipsparse_handle, nrow, ncol, &alpha, descrL, nnzL, Lx, Lp, Li, &beta, descrU, nnzU, Ui, Up,
        Ux, descrLU, LUx, LUp, LUi, &bufferSizeInBytes);

    hipsparseStatus_t const istat_bufferSizeExt = hipMalloc(&pBuffer, bufferSizeInBytes);

    // ------------------------------
    // estimate number of non-zeros
    // in L + U
    // ------------------------------
    hipsparseStatus_t const istat_Nnz = hipsparseXcsrgeam2Nnz(
        hipsparse_handle, nrow, ncol, descrL, nnzL, csrRowPtrL, csrColIndL, descrU, nnzU,
        csrRowPtrU, csrColIndU, descrLU, csrRowPtrLU, &nnzLU, pBuffer);

    HIP_CHECK(hipMalloc((void**)&csrColIndLU, sizeof(Iint) * nnzLU), ROCSOLVER_STATUS_ALLOC_FAILED);

    // -----------------------------------
    // allocate storage for csrValLU_array
    // -----------------------------------
    {
        HIP_CHECK(hipMalloc((void**)&(handle->csrValLU_array), sizeof(T*) * batch_count),
                  ROCSOLVER_STATUS_ALLOC_FAILED);
        if(handle->csrValLU_array == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
        {
            T* csrValLU = nullptr;
            size_t const nbytes = sizeof(T) * nnzLU;
            HIP_CHECK(hipMalloc((void**)&csrValLU, nbytes), ROCSOLVER_STATUS_ALLOC_FAILED);
            if(csrValLU == nullptr)
            {
                return (ROCSOLVER_STATUS_ALLOC_FAILED);
            };
            handle->csrValLU_array[ibatch] = csrValLU;
        };
    };

    handle->n = n;
    handle->nnzLU = nnzLU;
    // -------------------------------------------
    // Perform sparse matrix addition, LU = L + U
    // -------------------------------------------

    for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
    {
        T* csrValLU = handle->csrValLU_array[ibatch];

        HIPSPARSE_CHECK(hipsparseDcsrgeam2(hipsparse_handle, nrow, ncol, &alpha, descrL, nnzL,
                                           csrValL, csrRowPtrL, csrColIndL, &beta, descrU, nnzU,
                                           csrValU, csrRowPtrU, csrColIndU, descrLU, csrValLU,
                                           csrRowPtrLU, csrColIndLU, pBuffer),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);
    };

    HIP_CHECK(hipFree(pBuffer), ROCSOLVER_STATUS_INTERNAL_ERROR);
    pBuffer = nullptr;

    hipsparseStatus_t const istat_restore_pointer_mode
        = hipsparseSetPointerModel(hipsparse_handle, pointer_mode);
    if(istat_restore_pointer_mode != HIPSPARSE_STATUS_SUCCESS)
    {
        istat_return = (istat_return == ROCSOLVER_STATUS_SUCCESS) ? ROCSOLVER_STATUS_EXECUTION_FAILED
                                                                  : istat_return;
    };

    return (istat_return);
}

#endif
