
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
------------------------------------------------------------------------
This routine initializes the rocSolverRF library.  It allocates required
resources and must be called prior to any other rocSolverRF library
routine.
------------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfCreate(rocsolverRfHandle_t* p_handle)
{
    rocsolverRfHandle_t handle;

    {
        unsigned int const flags = hipHostMallocPortable;
        HIP_CHECK(hipHostMalloc((void**)&handle, sizeof(*handle), flags),
                  ROCSOLVER_STATUS_ALLOC_FAILED);

        if(handle == 0)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };
    };

    HIPSPARSE_CHECK(hipsparseCreate(&(handle->hipsparse_handle)), ROCSOLVER_STATUS_NOT_INITIALIZED);

    //--------------------
    //setup default values
    //--------------------
    handle->fast_mode = ROCSOLVERRF_RESET_VALUES_FAST_MODE_ON;
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
    handle->csrRowPtrL = 0;
    handle->csrColIndL = 0;
    handle->csrValL = 0;

    handle->nnzU = 0;
    handle->csrRowPtrU = 0;
    handle->csrColIndU = 0;
    handle->csrValU = 0;

    handle->nnzA = 0;
    handle->csrRowPtrA = 0;
    handle->csrColIndA = 0;

    handle->descrL = 0;
    handle->descrU = 0;
    handle->descrLU = 0;

    handle->infoL = 0;
    handle->infoU = 0;
    handle->infoLU_array = 0;

    handle->P_new2old = 0;
    handle->Q_new2old = 0;
    handle->Q_old2new = 0;

    handle->batch_count = 0;
    handle->csrValLU_array = 0;

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
