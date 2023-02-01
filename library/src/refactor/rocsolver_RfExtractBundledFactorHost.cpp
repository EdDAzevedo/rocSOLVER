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
    // ------------
    // check handle
    // ------------
    {
        bool const isok = (handle != nullptr) && (handle->hipsparse_handle != nullptr);
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

    int const n = handle->n;
    int const nnzLU = handle->nnzLU;
    size_t const nbytes_Mp = sizeof(int) * (n + 1);
    size_t const nbytes_Mi = sizeof(int) * nnzLU;
    size_t const nbytes_Mx = sizeof(double) * nnzLU;

    // ------------------------
    // allocate storage on host
    // ------------------------
    int* const Mp = (int*)malloc(nbytes_Mp);
    int* const Mi = (int*)malloc(nbytes_Mi);
    double* const Mx = (double*)malloc(nbytes_Mx);
    {
        bool const is_alloc_ok = (Mp != nullptr) && (Mi != nullptr) && (Mx != nullptr);
        if(!is_alloc_ok)
        {
            // -------------------------------------------
            // deallocate host memory to avoid memory leak
            // -------------------------------------------
            if(Mp != nullptr)
            {
                free(Mp);
            };
            if(Mi != nullptr)
            {
                free(Mi);
            };
            if(Mx != nullptr)
            {
                free(Mx);
            };
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };
    };

    {
        // ------------------------
        // copy from device to host
        // ------------------------
        int const ibatch = 0;

        hipError_t const istat_Mp = hipMemcpyDtoH(Mp, handle->csrRowPtrLU, nbytes_Mp);
        hipError_t const istat_Mi = hipMemcpyDtoH(Mi, handle->csrColIndLU, nbytes_Mi);
        hipError_t const istat_Mx = hipMemcpyDtoH(Mx, handle->csrValLU_array[ibatch], nbytes_Mx);

        bool const is_copy_ok
            = (istat_Mp == HIP_SUCCESS) && (istat_Mi == HIP_SUCCESS) && (istat_Mx == HIP_SUCCESS);
        if(!is_copy_ok)
        {
            // -------------------------------------------
            // deallocate host memory to avoid memory leak
            // -------------------------------------------
            if(Mp != nullptr)
            {
                free(Mp);
            };
            if(Mi != nullptr)
            {
                free(Mi);
            };
            if(Mx != nullptr)
            {
                free(Mx);
            };
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };
    };

    *h_nnzLU = nnzLU;
    *h_Mp = Mp;
    *h_Mi = Mi;
    *h_Mx = Mx;

    return (ROCSOLVER_STATUS_SUCCESS);
};
};
