
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
#include "hip_check.h"
#include "hipsparse/hipsparse.h"
#include "hipsparse_check.h"

extern "C" rocsolverStatus_t rocsolverRfSetupDevice(/* Input (in the device memory) */
                                                    int n,
                                                    int nnzA,
                                                    int* csrRowPtrA,
                                                    int* csrColIndA,
                                                    double* csrValA,
                                                    int nnzL,
                                                    int* csrRowPtrL,
                                                    int* csrColIndL,
                                                    double* csrValL,
                                                    int nnzU,
                                                    int* csrRowPtrU,
                                                    int* csrColIndU,
                                                    double* csrValU,
                                                    int* P,
                                                    int* Q,

                                                    /* Output */
                                                    rocsolverRfHandle_t handle)
{
    /*
     ------------
     Quick return
     ------------
     */
    if((n <= 0) || (nnzA <= 0) || (nnzL <= 0) || (nnzU <= 0))
    {
        return (ROCSOLVER_STATUS_SUCCESS);
    };

    {
        /*
    ---------------
    check arguments
    ---------------
    */
        bool const isok_A
            = (csrRowPtrA != nullptr) && (csrColIndA != nullptr) && (csrValA != nullptr);
        bool const isok_L
            = (csrRowPtrL != nullptr) && (csrColIndL != nullptr) && (csrValL != nullptr);
        bool const isok_U
            = (csrRowPtrU != nullptr) && (csrColIndU != nullptr) && (csrValU != nullptr);
        bool const isok_all = (handle != nullptr) && isok_A && isok_L && isok_U;
        if(!isok_all)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    handle->boost_val = 0;
    handle->effective_zero = 0;

    int* P_new2old = P;
    int* Q_new2old = Q;

    hipsparseHandle_t const hipsparse_handle = handle->hipsparse_handle;
    if(hipsparse_handle)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    /*
    ---------------------------------
    form sparsity pattern for (L + U)
    ---------------------------------
   */

    int* csrRowPtrLU = handle->csrRowPtrLU;
    int* csrColIndLU = handle->csrColIndLU;
    int* csrValLU = handle->csrValLU;
    hipsparseMatDescr_t descrLU = handle->descrLU;
    int nnz_LU = 0;

    if(csrRowPtrLU != nullptr)
    {
        HIP_CHECK(hipFree(csrRowPtrLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        csrRowPtrLU = nullptr;
    };

    if(csrColIndxLU != nullptr)
    {
        HIP_CHECK(hipFree(csrColIndLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        csrColIndxLU = nullptr;
    };

    if(csrValLU != nullptr)
    {
        HIP_CHECK(hipFree(csrValLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        csrValLU = nullptr;
    };

    if(descrLU != nullptr)
    {
        HIPSPARSE_CHECK(hipsparseMatDestroy(descrLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
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

    HIP_CHECK(hipMalloc((void**)&csrRowPtrLU, sizeof(int) * (n + 1)), ROCSOLVER_STATUS_ALLOC_ERROR);
    if(csrRowPtrLU == nullptr)
    {
        return (ROCSOLVER_STATUS_ALLOC_ERROR);
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

        HIPSPARSE_CHECK(hipsparsecsrgeam2_bufferSizeExt(
                            hipsparse_handle, nrow, ncol, &alpha, descrL, nnzL, csrValL, csrRowPtrL,
                            csrColIndL, &beta, descrU, nnzU, csrValU, csrRowPtrU, csrColIndU,
                            descrLU, csrValLU, csrRowPtrLU, csrColIndLU, &bufferSizeInBytes),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        HIP_CHECK(hipMalloc(&pBuffer, bufferSizeInBytes), ROCSOLVER_STATUS_ALLOC_ERROR);
        if(pBuffer == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_ERROR);
        };

        HIPSPARSE_CHECK(hipsparseXcsrgeam2Nnz(hipsparse_handle, nrow, ncol, descrL, nnzL,
                                              csrRowPtrL, csrColIndL, descrU, nnzU, csrRowPtrU,
                                              csrColIndU, descrLU, csrRowPtrLU, &nnz_LU, pBuffer),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        HIP_CHECK(hipMalloc((void**)&csrColIndLU, sizeof(int) * nnz_LU),
                  ROCSOLVER_STATUS_ALLOC_ERROR);
        if(csrColIndLU == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_ERROR);
        };

        HIP_CHECK(hipMalloc((void**)&csrValLU, sizeof(double) * nnz_LU),
                  ROCSOLVER_STATUS_ALLOC_ERROR);
        if(csrValLU == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_ERROR);
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
        handle->csrColLU = csrColLU;
    };

    /*
    ---------------------------------
    setup row and column permutations
    ---------------------------------
   */

    if(handle->P_new2old != nullptr)
    {
        HIP_CHECK(hipFree(handle->P_new2old), ROCSOLVER_STATUS_ALLOC_ERROR);
        handle->P_new2old = nullptr;
    };

    if(handle->Q_new2old != nullptr)
    {
        HIP_CHECK(hipFree(handle->Q_new2old), ROCSOLVER_STATUS_ALLOC_ERROR);
        handle->Q_new2old = nullptr;
    };

    if(handle->Q_old2new != nullptr)
    {
        HIP_CHECK(hipFree(handle->Q_old2new), ROCSOLVER_STATUS_ALLOC_ERROR);
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
        HIP_CHECK(hipMalloc(&(Q_old2new), sizeof(int) * n), ROCSOLVER_STATUS_ALLOC_ERROR);
        if(Q_old2new == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_ERROR);
        };

        hipStream_t streamId;
        HIPSPARSE_CHECK(hipsparseGetStream(handle->hipsparse_handle, &streamId),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);
        rocrefactor_ipvec(streamId, n, Q_new2old, Q_old2new);

        handle->Q_old2new = Q_old2new;
    }

    /*
  -----------------------------
  copy the values of A into L+U
  -----------------------------
  */

    rocsolverStatus_t istat
        = rocrefactor_RfResetValues(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle);

    return (istat);
}
