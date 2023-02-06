
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
#include "stdio.h"
#include "stdlib.h"

#include "hip_check.h"
#include "hipsparse_check.h"
#include "rocsolver_refactor.h"

/*
------------------------------------------------------------------------
This routine initializes the rocSolverRF library.  It allocates required
resources and must be called prior to any other rocSolverRF library
routine.
------------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfCreate(rocsolverRfHandle_t* p_handle)
{
    hipsparseHandle_t hipsparse_handle = nullptr;
    hipStream_t streamId = 0;

    rocsolverRfHandle_t handle = static_cast<rocsolverRfHandle_t>(malloc(sizeof(*handle)));

    bool const isok_handle = (handle != nullptr);

    hipError_t istat_streamId = hipStreamCreate(&streamId);
    bool const isok_streamId = (istat_streamId == HIP_SUCCESS);

    hipsparseStatus_t istat_hipsparse = hipsparseCreate(&hipsparse_handle);
    bool const isok_hipsparse = (istat_hipsparse == HIPSPARSE_STATUS_SUCCESS);

    bool const isok_alloc = isok_handle && isok_streamId && isok_hipsparse;

    bool const isok_set_stream = isok_alloc
        && (hipsparseSetStream(hipsparse_handle, streamId) == HIPSPARSE_STATUS_SUCCESS);

    bool const isok_all = (isok_alloc) && isok_set_stream;

    if(isok_all)
    {
        handle->hipsparse_handle = hipsparse_handle;
    }
    else
    {
        // -----------------------------
        // clean up to avoid memory leak
        // -----------------------------

        hipsparseDestroy(hipsparse_handle);
        hipStreamDestroy(streamId);
        free(handle);

        rocsolverStatus_t istat_return
            = (!isok_alloc) ? ROCSOLVER_STATUS_ALLOC_FAILED : ROCSOLVER_STATUS_INTERNAL_ERROR;
        return (istat_return);
    };

    //--------------------
    //setup default values
    //--------------------
    handle->fast_mode = ROCSOLVERRF_RESET_VALUES_FAST_MODE_OFF;
    handle->matrix_format = ROCSOLVERRF_MATRIX_FORMAT_CSR;
    handle->diag_format = ROCSOLVERRF_UNIT_DIAGONAL_ASSUMED_L;

    handle->numeric_boost = ROCSOLVERRF_NUMERIC_BOOST_NOT_USED;
    handle->boost_val = 0;
    handle->effective_zero = 0;

    // -----------------------------------------------------------------
    // note require  compatible algorithms for  factorization and solver
    // -----------------------------------------------------------------
    handle->fact_alg = ROCSOLVERRF_FACTORIZATION_ALG0;
    handle->solve_alg = ROCSOLVERRF_TRIANGULAR_SOLVE_ALG1;

    handle->nnzL = 0;
    handle->csrRowPtrL = nullptr;
    handle->csrColIndL = nullptr;
    handle->csrValL = nullptr;

    handle->nnzU = 0;
    handle->csrRowPtrU = nullptr;
    handle->csrColIndU = nullptr;
    handle->csrValU = nullptr;

    handle->nnzA = 0;
    handle->csrRowPtrA = nullptr;
    handle->csrColIndA = nullptr;

    handle->descrL = nullptr;
    handle->descrU = nullptr;
    handle->descrLU = nullptr;

    handle->infoL = nullptr;
    handle->infoU = nullptr;
    handle->infoLU_array = nullptr;

    handle->P_new2old = nullptr;
    handle->Q_new2old = nullptr;
    handle->Q_old2new = nullptr;

    handle->batch_count = 0;
    handle->csrValLU_array = nullptr;

    handle->n = 0;
    handle->nnzLU = 0;
    handle->csrRowPtrLU = 0;
    handle->csrColIndLU = 0;

    handle->buffer = 0;
    handle->buffer_size = 0;

    *p_handle = handle;
    return (ROCSOLVER_STATUS_SUCCESS);
};
};
