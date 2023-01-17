
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

rocsolverStatus_t cusolverRfBatchSetupHost(
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
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
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

    {
        // allocate and copy rowPtr  on GPU Device
        size_t nbytes_csrRowPtr = sizeof(int) * (n + 1);
        HIP_CHECK(hipMalloc(&d_csrRowPtrA, nbytes_csrRowPtr), ROCSOLVER_STATUS_ALLOC_FAILED);
        HIP_CHECK(hipMalloc(&d_csrRowPtrL, nbytes_csrRowPtr), ROCSOLVER_STATUS_ALLOC_FAILED);
        HIP_CHECK(hipMalloc(&d_csrRowPtrU, nbytes_csrRowPtr), ROCSOLVER_STATUS_ALLOC_FAILED);

        // copy rowPtr to device
        HIP_CHECK(hipMemcpyHtoD(d_csrRowPtrA, h_csrRowPtrA, nbytes_csrRowPtr),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);

        HIP_CHECK(hipMemcpyHtoD(d_csrRowPtrL, h_csrRowPtrL, nbytes_csrRowPtr),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);

        HIP_CHECK(hipMemcpyHtoD(d_csrRowPtrU, h_csrRowPtrU, nbytes_csrRowPtr),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);
    };

    {
        // allocate and copy ColInd on GPU Device
        size_t const nbytesA = sizeof(int) * nnzA;
        HIP_CHECK(hipMalloc(&d_csrColIndA, nbytesA), ROCSOLVER_STATUS_ALLOC_FAILED);
        HIP_CHECK(hipMemcpyHtoD(d_csrColIndA, h_csrColIndA, nbytesA),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);

        size_t const nbytesL = sizeof(int) * nnzL;
        HIP_CHECK(hipMalloc(&d_csrColIndL, nbytesL), ROCSOLVER_STATUS_ALLOC_FAILED);
        HIP_CHECK(hipMemcpyHtoD(d_csrColIndL, h_csrColIndL, nbytesL),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);

        size_t const nbytesU = sizeof(int) * nnzU;
        HIP_CHECK(hipMalloc(&d_csrColIndU, nbytesU), ROCSOLVER_STATUS_ALLOC_FAILED);
        HIP_CHECK(hipMemcpyHtoD(d_csrColIndU, h_csrColIndU, nbytesU),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);
    };

    {
        // allocate and copy Val

        size_t const nbytes_csrValA_array = sizeof(double*) * batch_count;

        HIP_CHECK(hipMalloc(&d_csrValA_array, nbytes_csrValA_array), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(d_csrValA_array == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            size_t const nbytesA = sizeof(double) * nnzA;

            double* d_csrValA = nullptr;
            HIP_CHECK(hipMalloc(&d_csrValA, nbytesA), ROCSOLVER_STATUS_ALLOC_FAILED);

            d_csrValA_array[ibatch] = d_csrValA;

            double* h_csrValA = h_csrValA_array[ibatch];

            HIP_CHECK(hipMemcpyHtoD(d_csrValA, h_csrValA, nbytesA),
                      ROCSOLVER_STATUS_EXECUTION_FAILED);
        };
        size_t const nbytesL = sizeof(double) * nnzL;
        HIP_CHECK(hipMalloc(&d_csrValL, nbytesL), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(d_csrValL == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        HIP_CHECK(hipMemcpyHtoD(d_csrValL, h_csrValL, nbytesL), ROCSOLVER_STATUS_EXECUTION_FAILED);

        size_t const nbytesU = sizeof(double) * nnzU;
        HIP_CHECK(hipMalloc(&d_csrValU, nbytesU), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(d_csrValU == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        HIP_CHECK(hipMemcpyHtoD(d_csrValU, h_csrValU, nbytesU), ROCSOLVER_STATUS_EXECUTION_FAILED);
    };

    int* d_P = nullptr;
    int* d_Q = nullptr;
    {
        size_t const nbytes_PQ = sizeof(int) * n;
        HIP_CHECK(hipMalloc(&d_P, nbytes_PQ), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(d_P == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        HIP_CHECK(hipMalloc(&d_Q, nbytes_PQ), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(d_Q == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        HIP_CHECK(hipMemcpyHtoD(d_P, h_P, nbytes_PQ), ROCSOLVER_STATUS_EXECUTION_FAILED);
        HIP_CHECK(hipMemcpyHtoD(d_Q, h_Q, nbytes_PQ), ROCSOLVER_STATUS_EXECUTION_FAILED);
    };

    bool constexpr MAKE_COPY = false;
    istat = rocsolverRfBatchSetupDevice_impl<MAKE_COPY>(
        batch_count, n, nnzA, d_csrRowPtrA, d_csrColIndA, d_csrValA_array, nnzL, d_csrRowPtrL,
        d_csrColIndL, d_csrValL, nnzU, d_csrRowPtrU, d_csrColIndU, d_csrValU, d_P, d_Q, handle);

    // cleanup  csrValA_array
    {
        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            if(d_csrValA_array[ibatch] != nullptr)
            {
                HIP_CHECK(hipFree(d_csrValA_array[ibatch]), ROCSOLVER_STATUS_INTERNAL_ERROR);
            };
            d_csrValA_array[ibatch] = nullptr;
        };

        HIP_CHECK(hipFree(d_csrValA_array), ROCSOLVER_STATUS_INTERNAL_ERROR);
        d_csrValA_array = nullptr;
    };

    return (istat);
};
};
