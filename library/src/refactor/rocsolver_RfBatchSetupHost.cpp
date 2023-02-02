
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
    {
        bool const isok_handle = (handle != nullptr) && (handle->hipsparse_handle != nullptr);
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

    int* d_csrRowPtrA = nullptr;
    int* d_csrRowPtrL = nullptr;
    int* d_csrRowPtrU = nullptr;

    int* d_csrColIndA = nullptr;
    int* d_csrColIndL = nullptr;
    int* d_csrColIndU = nullptr;

    double** d_csrValA_array = nullptr;

    double* d_csrValL = nullptr;
    double* d_csrValU = nullptr;

    int* d_P = nullptr;
    int* d_Q = nullptr;

    size_t const nbytes_PQ = sizeof(int) * n;

    size_t const nbytes_csrRowPtr = sizeof(int) * (n + 1);

    size_t const nbytes_csrColIndA = sizeof(int) * nnzA;
    size_t const nbytes_csrColIndL = sizeof(int) * nnzL;
    size_t const nbytes_csrColIndU = sizeof(int) * nnzU;

    size_t const nbytes_csrValL = sizeof(double) * nnzL;
    size_t const nbytes_csrValU = sizeof(double) * nnzU;
    size_t const nbytes_csrValA = sizeof(double) * nnzA;

    size_t const nbytes_csrValA_array = sizeof(double*) * batch_count;
    // ---------------------------
    // allocate all device storage
    // ---------------------------
    {
        hipError_t const istat_P = hipMalloc(&d_P, nbytes_PQ);
        hipError_t const istat_Q = hipMalloc(&d_Q, nbytes_PQ);
        bool const isok_PQ = (istat_P == HIP_SUCCESS) && (istat_Q == HIP_SUCCESS);

        hipError_t const istat_csrRowPtrA = hipMalloc(&d_csrRowPtrA, nbytes_csrRowPtr);
        hipError_t const istat_csrRowPtrL = hipMalloc(&d_csrRowPtrL, nbytes_csrRowPtr);
        hipError_t const istat_csrRowPtrU = hipMalloc(&d_csrRowPtrU, nbytes_csrRowPtr);
        bool const isok_csrRowPtr = (istat_csrRowPtrA == HIP_SUCCESS)
            && (istat_csrRowPtrL == HIP_SUCCESS) && (istat_csrRowPtrU == HIP_SUCCESS);

        hipError_t const istat_ColIndA = hipMalloc(&d_csrColIndA, nbytes_csrColIndA);
        hipError_t const istat_ColIndL = hipMalloc(&d_csrColIndL, nbytes_csrColIndL);
        hipError_t const istat_ColIndU = hipMalloc(&d_csrColIndU, nbytes_csrColIndU);
        bool const isok_ColInd = (istat_ColIndA == HIP_SUCCESS) && (istat_ColIndL == HIP_SUCCESS)
            && (istat_ColIndU == HIP_SUCCESS);

        hipError_t const istat_csrValL = hipMalloc(&d_csrValL, nbytes_csrValL);
        hipError_t const istat_csrValU = hipMalloc(&d_csrValU, nbytes_csrValU);
        bool const isok_csrVal = (istat_csrValL == HIP_SUCCESS) && (istat_csrValU == HIP_SUCCESS);

        d_csrValA_array = (double**)malloc(nbytes_csrValA_array);
        bool const isok_csrValA_array = (d_csrValA_array == nullptr);

        bool isok_batch = true;
        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            hipError_t const istat = hipMalloc(&(d_csrValA_array[ibatch]), nbytes_csrValA);
            bool const isok = (istat == HIP_SUCCESS);
            isok_batch = isok_batch && isok;
        };

        bool const isok_alloc = isok_PQ && isok_batch && isok_csrVal && isok_ColInd
            && isok_csrRowPtr && isok_csrValA_array;
        if(!isok_alloc)
        {
            // ---------------------------------------
            // deallocate storage to avoid memory leak
            // ---------------------------------------
            hipFree(d_P);
            hipFree(d_Q);

            hipFree(d_csrRowPtrA);
            hipFree(d_csrRowPtrL);
            hipFree(d_csrRowPtrU);

            hipFree(d_csrColIndA);
            hipFree(d_csrColIndL);
            hipFree(d_csrColIndU);

            hipFree(d_csrValL);
            hipFree(d_csrValU);

            for(int ibatch = 0; ibatch < batch_count; ibatch++)
            {
                hipFree(d_csrValA_array[ibatch]);
            };
            free(d_csrValA_array);

            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };
    };

    // -------------------------------
    // copy arrays from host to device
    // -------------------------------
    {
        hipError_t const istat_P = hipMemcpyHtoD(d_P, h_P, nbytes_PQ);
        hipError_t const istat_Q = hipMemcpyHtoD(d_Q, h_Q, nbytes_PQ);
        bool const isok_PQ = (istat_P == HIP_SUCCESS) && (istat_Q == HIP_SUCCESS);

        hipError_t const istat_csrRowPtrA
            = hipMemcpyHtoD(d_csrRowPtrA, h_csrRowPtrA, nbytes_csrRowPtr);
        hipError_t const istat_csrRowPtrL
            = hipMemcpyHtoD(d_csrRowPtrL, h_csrRowPtrL, nbytes_csrRowPtr);
        hipError_t const istat_csrRowPtrU
            = hipMemcpyHtoD(d_csrRowPtrU, h_csrRowPtrU, nbytes_csrRowPtr);
        bool const isok_csrRowPtr = (istat_csrRowPtrA == HIP_SUCCESS)
            && (istat_csrRowPtrL == HIP_SUCCESS) && (istat_csrRowPtrU == HIP_SUCCESS);

        hipError_t const istat_csrColIndA
            = hipMemcpyHtoD(d_csrColIndA, h_csrColIndA, nbytes_csrColIndA);
        hipError_t const istat_csrColIndL
            = hipMemcpyHtoD(d_csrColIndL, h_csrColIndL, nbytes_csrColIndL);
        hipError_t const istat_csrColIndU
            = hipMemcpyHtoD(d_csrColIndU, h_csrColIndU, nbytes_csrColIndU);
        bool const isok_csrColInd = (istat_csrColIndA == HIP_SUCCESS)
            && (istat_csrColIndL == HIP_SUCCESS) && (istat_csrColIndU == HIP_SUCCESS);

        bool isok_csrValA = true;
        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            hipError_t const istat_csrValA
                = hipMemcpyHtoD(d_csrValA_array[ibatch], h_csrValA_array[ibatch], nbytes_csrValA);
            bool const isok = (istat_csrValA == HIP_SUCCESS);
            isok_csrValA = isok_csrValA && isok;
        };

        bool const isok_HtoD = isok_PQ && isok_csrValA && isok_csrRowPtr && isok_csrColInd;
        if(!isok_HtoD)
        {
            hipFree(d_P);
            hipFree(d_Q);

            hipFree(d_csrRowPtrA);
            hipFree(d_csrRowPtrL);
            hipFree(d_csrRowPtrU);

            hipFree(d_csrColIndA);
            hipFree(d_csrColIndL);
            hipFree(d_csrColIndU);

            hipFree(d_csrValL);
            hipFree(d_csrValU);

            for(int ibatch = 0; ibatch < batch_count; ibatch++)
            {
                hipFree(d_csrValA_array[ibatch]);
            };
            free(d_csrValA_array);

            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };
    };

    bool constexpr MAKE_COPY = false;
    rocsolverStatus_t const istat_setup = rocsolverRfBatchSetupDevice_impl<MAKE_COPY>(
        batch_count, n, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA_array, nnzL, d_csrRowPtrL,
        d_csrColIndL, d_csrValL, nnzU, d_csrRowPtrU, d_csrColIndU, d_csrValU, d_P, d_Q, handle);

    return (istat_setup);
};
};
