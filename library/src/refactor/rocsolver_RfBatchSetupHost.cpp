
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

#include "rocsolver_RfBatchSetupDevice.hpp"

/*
----------------------------------------------------------------------
This routine assembles the internal data structures of the rocSolverRF
library.  It is often the first routine to be called after the call to
the rocsolverRfCreate() routine.

This routine accepts as input (on the device) the original matrix A,
the lower L and upper U triangular factors, as well as the left (P)
and right (Q) permutations resulting from the full LU factorization of
the first (i=1) linear system

   A_i x_i = f_i

The permutations P and Q represent the final composition of all the left
and right reordering applied to the original matrix A, respectively.
However, these permutations are often associated with partial pivoting
and reordering to minimize fill-in, respectively.

This routine needs to be called only for a single linear system

  
   A_i x_i = f_i


----------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfBatchSetupHost(
    /* Input (in the host memory) */
    int batch_count,
    int n,
    int nnzA,
    int* h_csrRowPtrA,
    int* h_csrColIndA,
    double* h_csrValA_array[],
    int nnzL,
    int* h_csrRowPtrL,
    int* h_csrColIndL,
    double* h_csrValL,
    int nnzU,
    int* h_csrRowPtrU,
    int* h_csrColIndU,
    double* h_csrValU,
    int* h_P,
    int* h_Q,
    /* Output */
    rocsolverRfHandle_t handle)
{
    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;

    // ---------------
    // check arguments
    // ---------------
    {
        bool const isok_handle = (handle != nullptr) && (handle->hipsparse_handle.data() != nullptr);
        if(!isok_handle)
        {
            return (ROCSOLVER_STATUS_NOT_INITIALIZED);
        };
    };

    if(h_csrValA_array == nullptr)
    {
        return (ROCSOLVER_STATUS_INVALID_VALUE);
    };

    rocsolverStatus_t istat = rocsolver_RfBatchSetup_checkargs(
        batch_count, n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA_array, nnzL, h_csrRowPtrL,
        h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, h_P, h_Q, handle);

    if(istat != ROCSOLVER_STATUS_SUCCESS)
    {
        return (istat);
    };

    try
    {
        istat_return
            = rocsolverRfBatchSetupDevice_impl(/* Input (in the device memory) */
                                               batch_count, n, nnzA, h_csrRowPtrA, h_csrColIndA,
                                               h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL,
                                               h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU,
                                               h_csrValU, h_P, h_Q, handle);
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
