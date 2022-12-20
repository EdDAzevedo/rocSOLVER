
/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rf_common.hpp"

#include "rocsolver_RfResetValues.hpp"
#include "rocsolver_ipvec.hpp"

template <typename T>
rocsolverStatus_t rocsolver_RfSetup_checkargs(int n,
                                              int nnzA,
                                              int* csrRowPtrA,
                                              int* csrColIndA,
                                              T* csrValA,
                                              int nnzL,
                                              int* csrRowPtrL,
                                              int* csrColIndL,
                                              T* csrValL,
                                              int nnzU,
                                              int* csrRowPtrU,
                                              int* csrColIndU,
                                              T* csrValU,
                                              int* P,
                                              int* Q,
                                              rocsolverRfHandle_t handle)
{
    // ---------------
    // check arguments
    // ---------------
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    hipsparseHandle_t const hipsparse_handle = handle->hipsparse_handle;
    if(hipsparse_handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    bool const isok_scalar = (n >= 0) && (nnzA >= 0) && (nnzL >= 0) && (nnzU >= 0);
    bool const isok_A = (csrRowPtrA != nullptr) && (csrColIndA != nullptr) && (csrValA != nullptr);
    bool const isok_L = (csrRowPtrL != nullptr) && (csrColIndL != nullptr) && (csrValL != nullptr);
    bool const isok_U = (csrRowPtrU != nullptr) && (csrColIndU != nullptr) && (csrValU != nullptr);
    bool const isok_all = isok_scalar && isok_A && isok_L && isok_U;
    if(!isok_all)
    {
        return (ROCSOLVER_STATUS_INVALID_VALUE);
    };

    /*
     ------------
     Quick return
     ------------
     */
    if((n == 0) || (nnzA == 0) || (nnzL == 0) || (nnzU == 0))
    {
        return (ROCSOLVER_STATUS_SUCCESS);
    };

    return (ROCSOLVER_STATUS_SUCCESS);
};

template <bool MAKE_COPY>
rocsolverStatus_t rocsolverRfSetupDevice_impl(/* Input (in the device memory) */
                                              int n,
                                              int nnzA,
                                              int* csrRowPtrA_in,
                                              int* csrColIndA_in,
                                              double* csrValA_in,
                                              int nnzL,
                                              int* csrRowPtrL_in,
                                              int* csrColIndL_in,
                                              double* csrValL_in,
                                              int nnzU,
                                              int* csrRowPtrU_in,
                                              int* csrColIndU_in,
                                              double* csrValU_in,
                                              int* P_in,
                                              int* Q_in,

                                              /* Output */
                                              rocsolverRfHandle_t handle)
{
    // check args
    rocsolverStatus_t istat = rocsolver_RfSetup_checkargs(
        n, nnzA, csrRowPtrA_in, csrColIndA_in, csrValA_in, nnzL, csrRowPtrL_in, csrColIndL_in,
        csrValL_in, nnzU, csrRowPtrU_in, csrColIndU_in, csrValU_in, P_in, Q_in, handle);
    if(istat != ROCSOLVER_STATUS_SUCCESS)
    {
        return (istat);
    };

    hipsparseHandle_t const hipsparse_handle = handle->hipsparse_handle;
    handle->boost_val = 0;
    handle->effective_zero = 0;

    int* csrRowPtrA = csrRowPtrA_in;
    int* csrColIndA = csrColIndA_in;
    double* csrValA = csrValA_in;

    int* csrRowPtrL = csrRowPtrL_in;
    int* csrColIndL = csrColIndL_in;
    double* csrValL = csrValL_in;

    int* csrRowPtrU = csrRowPtrU_in;
    int* csrColIndU = csrColIndU_in;
    double* csrValU = csrValU_in;

    int* P = P_in;
    int* Q = Q_in;

    if(MAKE_COPY)
    {
        // allocate and copy P, Q
        size_t const nbytes_PQ = sizeof(int) * n;

        HIP_CHECK(hipMalloc(&P, nbytes_PQ), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&Q, nbytes_PQ), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMemcpyDtoD(P, P_in, nbytes_PQ), ROCSOLVER_STATUS_EXECUTION_FAILED);

        HIP_CHECK(hipMemcpyDtoD(Q, Q_in, nbytes_PQ), ROCSOLVER_STATUS_EXECUTION_FAILED);

        // allocate and copy A
        size_t const nbytes_RowPtrA = sizeof(int) * (n + 1);
        size_t const nbytes_ColIndA = sizeof(int) * nnzA;
        size_t const nbytes_ValA = sizeof(double) * nnzA;

        HIP_CHECK(hipMalloc(&csrRowPtrA, nbytes_RowPtrA), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&csrColIndA, nbytes_ColIndA), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&csrValA, nbytes_ValA), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMemcpyDtoD(csrRowPtrA, csrRowPtrA_in, nbytes_RowPtrA),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);
        HIP_CHECK(hipMemcpyDtoD(csrColIndA, csrColIndA_in, nbytes_ColIndA),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);
        HIP_CHECK(hipMemcpyDtoD(csrValA, csrValA_in, nbytes_ValA), ROCSOLVER_STATUS_EXECUTION_FAILED);

        // allocate and copy L
        size_t const nbytes_RowPtrL = sizeof(int) * (n + 1);
        size_t const nbytes_ColIndL = sizeof(int) * nnzL;
        size_t const nbytes_ValL = sizeof(double) * nnzL;

        HIP_CHECK(hipMalloc(&csrRowPtrL, nbytes_RowPtrL), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&csrColIndL, nbytes_ColIndL), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&csrValL, nbytes_ValL), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMemcpyDtoD(csrRowPtrL, csrRowPtrL_in, nbytes_RowPtrL),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);
        HIP_CHECK(hipMemcpyDtoD(csrColIndL, csrColIndL_in, nbytes_ColIndL),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);
        HIP_CHECK(hipMemcpyDtoD(csrValL, csrValL_in, nbytes_ValL), ROCSOLVER_STATUS_EXECUTION_FAILED);

        // allocate and copy U
        size_t const nbytes_RowPtrU = sizeof(int) * (n + 1);
        size_t const nbytes_ColIndU = sizeof(int) * nnzU;
        size_t const nbytes_ValU = sizeof(double) * nnzU;

        HIP_CHECK(hipMalloc(&csrRowPtrU, nbytes_RowPtrU), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&csrColIndU, nbytes_ColIndU), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&csrValU, nbytes_ValU), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMemcpyDtoD(csrRowPtrU, csrRowPtrU_in, nbytes_RowPtrU),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);
        HIP_CHECK(hipMemcpyDtoD(csrColIndU, csrColIndU_in, nbytes_ColIndU),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);
        HIP_CHECK(hipMemcpyDtoD(csrValU, csrValU_in, nbytes_ValU), ROCSOLVER_STATUS_EXECUTION_FAILED);
    };

    int* P_new2old = P;
    int* Q_new2old = Q;

    /*
    ---------------------------------
    form sparsity pattern for (L + U)
    ---------------------------------
   */

    int* csrRowPtrLU = handle->csrRowPtrLU;
    int* csrColIndLU = handle->csrColIndLU;
    double* csrValLU = handle->csrValLU;
    hipsparseMatDescr_t descrLU = handle->descrLU;
    int nnz_LU = 0;

    if(csrRowPtrLU != nullptr)
    {
        HIP_CHECK(hipFree(csrRowPtrLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        csrRowPtrLU = nullptr;
    };

    if(csrColIndLU != nullptr)
    {
        HIP_CHECK(hipFree(csrColIndLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        csrColIndLU = nullptr;
    };

    if(csrValLU != nullptr)
    {
        HIP_CHECK(hipFree(csrValLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        csrValLU = nullptr;
    };

    if(descrLU != nullptr)
    {
        HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        descrLU = nullptr;
    };

    hipsparseMatDescr_t descrL;
    hipsparseMatDescr_t descrU;
    csrsv2Info_t infoL;
    csrsv2Info_t infoU;

    {
        /*
   -------------
   setup descrL
   -------------
  */
        HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrL), ROCSOLVER_STATUS_INTERNAL_ERROR);
        HIPSPARSE_CHECK(hipsparseSetMatType(descrL, HIPSPARSE_MATRIX_TYPE_GENERAL),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);
        HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descrL, HIPSPARSE_INDEX_BASE_ZERO),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);
        HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&infoL), ROCSOLVER_STATUS_INTERNAL_ERROR);

        /*
   -------------
   setup descrU
   -------------
  */
        HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        HIPSPARSE_CHECK(hipsparseSetMatType(descrU, HIPSPARSE_MATRIX_TYPE_GENERAL),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);
        HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descrU, HIPSPARSE_INDEX_BASE_ZERO),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);
        HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&infoU), ROCSOLVER_STATUS_INTERNAL_ERROR);

        /*
   -------------
   setup descrLU
   -------------
  */
        HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        HIPSPARSE_CHECK(hipsparseSetMatType(descrLU, HIPSPARSE_MATRIX_TYPE_GENERAL),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);
        HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descrLU, HIPSPARSE_INDEX_BASE_ZERO),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);
        HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&infoL), ROCSOLVER_STATUS_INTERNAL_ERROR);
    }

    HIP_CHECK(hipMalloc((void**)&csrRowPtrLU, sizeof(int) * (n + 1)), ROCSOLVER_STATUS_ALLOC_FAILED);
    if(csrRowPtrLU == nullptr)
    {
        return (ROCSOLVER_STATUS_ALLOC_FAILED);
    };

    {
        int nrow = n;
        int ncol = n;

        double alpha = 1;
        double beta = 1;

        HIPSPARSE_CHECK(hipsparseSetPointerMode(hipsparse_handle, HIPSPARSE_POINTER_MODE_HOST),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        void* pBuffer = nullptr;
        size_t bufferSizeInBytes = 0;

        HIPSPARSE_CHECK(hipsparseDcsrgeam2_bufferSizeExt(
                            hipsparse_handle, nrow, ncol, &alpha, descrL, nnzL, csrValL, csrRowPtrL,
                            csrColIndL, &beta, descrU, nnzU, csrValU, csrRowPtrU, csrColIndU,
                            descrLU, csrValLU, csrRowPtrLU, csrColIndLU, &bufferSizeInBytes),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        HIP_CHECK(hipMalloc(&pBuffer, bufferSizeInBytes), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(pBuffer == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        HIPSPARSE_CHECK(hipsparseXcsrgeam2Nnz(hipsparse_handle, nrow, ncol, descrL, nnzL,
                                              csrRowPtrL, csrColIndL, descrU, nnzU, csrRowPtrU,
                                              csrColIndU, descrLU, csrRowPtrLU, &nnz_LU, pBuffer),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        HIP_CHECK(hipMalloc((void**)&csrColIndLU, sizeof(int) * nnz_LU),
                  ROCSOLVER_STATUS_ALLOC_FAILED);
        if(csrColIndLU == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        HIP_CHECK(hipMalloc((void**)&csrValLU, sizeof(double) * nnz_LU),
                  ROCSOLVER_STATUS_ALLOC_FAILED);
        if(csrValLU == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        handle->n = n;
        handle->nnz_LU = nnz_LU;
        /*
     ----------
     Perform LU = L + U
     ----------
     */

        HIPSPARSE_CHECK(hipsparseDcsrgeam2(hipsparse_handle, nrow, ncol, &alpha, descrL, nnzL,
                                           csrValL, csrRowPtrL, csrColIndL, &beta, descrU, nnzU,
                                           csrValU, csrRowPtrU, csrColIndU, descrLU, csrValLU,
                                           csrRowPtrLU, csrColIndLU, pBuffer),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        HIP_CHECK(hipFree(pBuffer), ROCSOLVER_STATUS_INTERNAL_ERROR);
        pBuffer = nullptr;

        HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrL), ROCSOLVER_STATUS_INTERNAL_ERROR);
        HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descrU), ROCSOLVER_STATUS_INTERNAL_ERROR);

        handle->descrLU = descrLU;
        handle->csrValLU = csrValLU;
        handle->csrRowPtrLU = csrRowPtrLU;
        handle->csrColIndLU = csrColIndLU;
    };

    /*
    ---------------------------------
    setup row and column permutations
    ---------------------------------
   */

    if(handle->P_new2old != nullptr)
    {
        HIP_CHECK(hipFree(handle->P_new2old), ROCSOLVER_STATUS_ALLOC_FAILED);
        handle->P_new2old = nullptr;
    };

    if(handle->Q_new2old != nullptr)
    {
        HIP_CHECK(hipFree(handle->Q_new2old), ROCSOLVER_STATUS_ALLOC_FAILED);
        handle->Q_new2old = nullptr;
    };

    if(handle->Q_old2new != nullptr)
    {
        HIP_CHECK(hipFree(handle->Q_old2new), ROCSOLVER_STATUS_ALLOC_FAILED);
        handle->Q_old2new = nullptr;
    };

    handle->P_new2old = P_new2old;
    handle->Q_new2old = Q_new2old;

    {
        /*
    -------------------------
    Create inverse permutation Q_old2new[]

    inew = Q_new2old[ iold ]
    Q_old2new[ iold ] = inew;
    -------------------------
    */
        int* Q_old2new = nullptr;
        HIP_CHECK(hipMalloc(&(Q_old2new), sizeof(int) * n), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(Q_old2new == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        hipStream_t streamId;
        HIPSPARSE_CHECK(hipsparseGetStream(handle->hipsparse_handle, &streamId),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);
        rocsolver_ipvec_template(streamId, n, Q_new2old, Q_old2new);

        handle->Q_old2new = Q_old2new;
    }

    /*
  -----------------------------
  copy the values of A into L+U
  -----------------------------
  */

    return (rocsolver_RfResetValues_template<int, int, double>(n, nnzA, csrRowPtrA, csrColIndA,
                                                               csrValA, P, Q, handle));
}
