/*! \file */
/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <stdlib.h>

/*
----------------------------------------------------------------------
This routine extracts lower (L) and upper (U) triangular factors from
the rocSovlerRF library handle into the host memory.  The factors
are compressed into a single matrix M = (L-I)+U, where the unitary
diagonal of (L) is not stored.  It is assumed that a prior call to the
rocsolverRfRefactor() was called to generate the triangular factors.


Host memory is allocated then the M = (L-I) + U factors on device are copied
into Host memory
----------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfExtractBundledFactorsHost(rocsolverRfHandle_t handle,
                                                       /* Output in host memory */
                                                       int* h_nnzLU,
                                                       int** h_Mp,
                                                       int** h_Mi,
                                                       double** h_Mx)
{
    int const idebug = 1;

    // ------------
    // check handle
    // ------------
    {
        bool const isok = (handle != nullptr);
        if(!isok)
        {
            return (ROCSOLVER_STATUS_NOT_INITIALIZED);
        };
    };

    // ---------------
    // check arguments
    // ---------------
    {
        bool const isok
            = (h_nnzLU != nullptr) && (h_Mp != nullptr) && (h_Mi != nullptr) && (h_Mx != nullptr);
        if(!isok)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;

    if(idebug >= 1)
    {
        printf("%s:%d, istat_return=%d\n", __FILE__, __LINE__, istat_return);
        fflush(stdout);
    };

    try
    {
        int const n = handle->n;
        int const nnzLU = handle->nnzLU;

        int *Mp =  *h_Mp;
        int *Mi =  *h_Mi;
        double *Mx =  *h_Mx;


        // ------------------------
        // allocate storage on host
        // ------------------------
        bool const need_alloc_Mp = (Mp == nullptr);
        bool const need_alloc_Mi = (Mi == nullptr);
        bool const need_alloc_Mx = (Mx == nullptr);

        if (need_alloc_Mp) { Mp = new int[n+1]; };
        if (need_alloc_Mi) { Mi = new int[nnzLU]; };
        if (need_alloc_Mx) { Mx = new double[nnzLU]; };


        if(idebug >= 1)
        {
            printf("%s:%d, istat_return=%d\n", __FILE__, __LINE__, istat_return);
            fflush(stdout);
        };

        // ------------------------
        // copy from device to host
        // ------------------------

        // int* d_Mp = handle->csrRowPtrLU.data().get();
        // thrust::copy(d_Mp, d_Mp + (n + 1), Mp);
        thrust::copy(handle->csrRowPtrLU.begin(), handle->csrRowPtrLU.end(), Mp);
        if(idebug >= 1)
        {
            printf("%s:%d, istat_return=%d\n", __FILE__, __LINE__, istat_return);
            fflush(stdout);
        };

        // int* d_Mi = handle->csrColIndLU.data().get();
        // thrust::copy(d_Mi, d_Mi + nnzLU, Mi);
        thrust::copy(handle->csrColIndLU.begin(), handle->csrColIndLU.end(), Mi);
        if(idebug >= 1)
        {
            printf("%s:%d, istat_return=%d\n", __FILE__, __LINE__, istat_return);
            fflush(stdout);
        };

        // double* d_Mx = handle->csrValLU_array.data().get();
        // thrust::copy(d_Mx, d_Mx + nnzLU, Mx);
        thrust::copy(handle->csrValLU_array.begin(), handle->csrValLU_array.begin() + nnzLU, Mx);
        if(idebug >= 1)
        {
            printf("%s:%d, istat_return=%d\n", __FILE__, __LINE__, istat_return);
            fflush(stdout);
        };

        *h_Mp = Mp;
        *h_Mi = Mi;
        *h_Mx = Mx;
        *h_nnzLU = nnzLU;
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
};
};
