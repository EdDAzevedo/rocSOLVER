
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
rocsolverStatus_t rocsolver_RfBatchSetup_checkargs(int batch_count,
                                                   int n,
                                                   int nnzA,
                                                   int* csrRowPtrA,
                                                   int* csrColIndA,
                                                   T* csrValA_array[],
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

    bool const isok_scalar
        = (n >= 0) && (nnzA >= 0) && (nnzL >= 0) && (nnzU >= 0) && (batch_count >= 0);
    bool const isok_A
        = (csrRowPtrA != nullptr) && (csrColIndA != nullptr) && (csrValA_array != nullptr);
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
    if((n == 0) || (nnzA == 0) || (nnzL == 0) || (nnzU == 0) || (batch_count == 0))
    {
        return (ROCSOLVER_STATUS_SUCCESS);
    };

    return (ROCSOLVER_STATUS_SUCCESS);
};

template <bool MAKE_COPY>
rocsolverStatus_t rocsolverRfBatchSetupDevice_impl(/* Input (in the device memory) */
                                                   int batch_count,
                                                   int n,
                                                   int nnzA,
                                                   int* csrRowPtrA_in,
                                                   int* csrColIndA_in,
                                                   double* csrValA_array_in[],
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
    rocsolverStatus_t istat = rocsolver_RfBatchSetup_checkargs(
        batch_count, n, nnzA, csrRowPtrA_in, csrColIndA_in, csrValA_array_in, nnzL, csrRowPtrL_in,
        csrColIndL_in, csrValL_in, nnzU, csrRowPtrU_in, csrColIndU_in, csrValU_in, P_in, Q_in,
        handle);
    if(istat != ROCSOLVER_STATUS_SUCCESS)
    {
        return (istat);
    };

    hipsparseHandle_t const hipsparse_handle = handle->hipsparse_handle;
    handle->boost_val = 0;
    handle->effective_zero = 0;

    int* csrRowPtrA = csrRowPtrA_in;
    int* csrColIndA = csrColIndA_in;
    double** csrValA_array = csrValA_array_in;

    int* csrRowPtrL = csrRowPtrL_in;
    int* csrColIndL = csrColIndL_in;
    double* csrValL = csrValL_in;

    int* csrRowPtrU = csrRowPtrU_in;
    int* csrColIndU = csrColIndU_in;
    double* csrValU = csrValU_in;

    int* P = P_in;
    int* Q = Q_in;

    {
        // --------------------------------
        // setup storage for csrValLU_array
        // --------------------------------
        handle->batch_count = batch_count;
        if(handle->csrValLU_array != nullptr)
        {
            for(int ibatch = 0; ibatch < batch_count; ibatch++)
            {
                if(handle->csrValLU_array[ibatch] != nullptr)
                {
                    HIP_CHECK(hipFree(handle->csrValLU_array[ibatch]),
                              ROCSOLVER_STATUS_INTERNAL_ERROR);
                    handle->csrValLU_array[ibatch] = nullptr;
                };
            };

            HIP_CHECK(hipFree(handle->csrValLU_array), ROCSOLVER_STATUS_INTERNAL_ERROR);
            handle->csrValLU_array = nullptr;
        };

        HIP_CHECK(hipHostMalloc(&(handle->csrValLU_array), sizeof(double*) * batch_count,
                                hipHostMallocPortable),
                  ROCSOLVER_STATUS_ALLOC_FAILED);
        if(handle->csrValLU_array == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            double* csrValLU = nullptr;
            size_t const nbytes_L = sizeof(double) * nnzL;
            size_t const nbytes_U = sizeof(double) * nnzU;
            size_t const nbytes_LU = nbytes_L + nbytes_U;
            HIP_CHECK(hipMalloc(&csrValLU, nbytes_LU), ROCSOLVER_STATUS_ALLOC_FAILED);
            if(csrValLU == nullptr)
            {
                return (ROCSOLVER_STATUS_ALLOC_FAILED);
            };

            handle->csrValLU_array[ibatch] = csrValLU;
            {
                void* dst = csrValLU;
                int const value = 0;
                size_t const sizeBytes = nbytes_LU;
                HIP_CHECK(hipMemset(dst, value, sizeBytes), ROCSOLVER_STATUS_EXECUTION_FAILED);
            };
        };
    };

    if(MAKE_COPY)
    {
        // allocate and copy P, Q
        size_t const nbytes_PQ = sizeof(int) * n;

        HIP_CHECK(hipMalloc(&P, nbytes_PQ), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(P == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        HIP_CHECK(hipMalloc(&Q, nbytes_PQ), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(Q == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        HIP_CHECK(hipMemcpyDtoD(P, P_in, nbytes_PQ), ROCSOLVER_STATUS_EXECUTION_FAILED);

        HIP_CHECK(hipMemcpyDtoD(Q, Q_in, nbytes_PQ), ROCSOLVER_STATUS_EXECUTION_FAILED);

        // allocate and copy A
        size_t const nbytes_RowPtrA = sizeof(int) * (n + 1);
        size_t const nbytes_ColIndA = sizeof(int) * nnzA;

        HIP_CHECK(hipMalloc(&csrRowPtrA, nbytes_RowPtrA), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(csrRowPtrA == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        HIP_CHECK(hipMalloc(&csrColIndA, nbytes_ColIndA), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(csrColIndA == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        HIP_CHECK(hipMemcpyDtoD(csrRowPtrA, csrRowPtrA_in, nbytes_RowPtrA),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);
        HIP_CHECK(hipMemcpyDtoD(csrColIndA, csrColIndA_in, nbytes_ColIndA),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);

        // -------------------------------
        // allocate and copy csrValA_array
        // -------------------------------
        {
            size_t const nbytes_csrValA_array = sizeof(double*) * batch_count;
            HIP_CHECK(hipMalloc(&csrValA_array, nbytes_csrValA_array), ROCSOLVER_STATUS_ALLOC_FAILED);
            if(csrValA_array == nullptr)
            {
                return (ROCSOLVER_STATUS_ALLOC_FAILED);
            };

            for(int ibatch = 0; ibatch < batch_count; ibatch++)
            {
                double* csrValA = nullptr;
                size_t const nbytes_ValA = nnzA * sizeof(double);

                HIP_CHECK(hipMalloc(&csrValA, nbytes_ValA), ROCSOLVER_STATUS_ALLOC_FAILED);
                if(csrValA == nullptr)
                {
                    return (ROCSOLVER_STATUS_ALLOC_FAILED);
                };
                csrValA_array[ibatch] = csrValA;

                double* csrValA_in = csrValA_array_in[ibatch];
                HIP_CHECK(hipMemcpyDtoD(csrValA, csrValA_in, nbytes_ValA),
                          ROCSOLVER_STATUS_EXECUTION_FAILED);
            };
        };
        // -------------------
        // allocate and copy L
        // -------------------
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

    int const ibatch = 0;
    double* csrValLU = handle->csrValLU_array[ibatch];

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

    hipsparseMatDescr_t descrL = handle->descrL;
    hipsparseMatDescr_t descrU = handle->descrU;
    csrsv2Info_t infoL = handle->infoL;
    csrsv2Info_t infoU = handle->infoU;

    {
        // -------------
        // setup descrL and infoL
        // -------------
        if(descrL == nullptr)
        {
            HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrL), ROCSOLVER_STATUS_INTERNAL_ERROR);
            HIPSPARSE_CHECK(hipsparseSetMatType(descrL, HIPSPARSE_MATRIX_TYPE_GENERAL),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
            HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descrL, HIPSPARSE_INDEX_BASE_ZERO),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
        };
        if(infoL == nullptr)
        {
            HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&infoL), ROCSOLVER_STATUS_INTERNAL_ERROR);
        };

        // -------------
        // setup descrU and infoU
        // -------------
        if(descrU == nullptr)
        {
            HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrU), ROCSOLVER_STATUS_INTERNAL_ERROR);
            HIPSPARSE_CHECK(hipsparseSetMatType(descrU, HIPSPARSE_MATRIX_TYPE_GENERAL),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
            HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descrU, HIPSPARSE_INDEX_BASE_ZERO),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
        };
        if(infoU == nullptr)
        {
            HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&infoU), ROCSOLVER_STATUS_INTERNAL_ERROR);
        };

        // -------------
        // setup descrLU and infoLU_array
        // -------------
        if(descrLU == nullptr)
        {
            HIPSPARSE_CHECK(hipsparseCreateMatDescr(&descrLU), ROCSOLVER_STATUS_INTERNAL_ERROR);
            HIPSPARSE_CHECK(hipsparseSetMatType(descrLU, HIPSPARSE_MATRIX_TYPE_GENERAL),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
            HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descrLU, HIPSPARSE_INDEX_BASE_ZERO),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
        };

        if(handle->infoLU_array == nullptr)
        {
            size_t const nbytes = batch_count * sizeof(csrilu02Info_t);
            HIP_CHECK(hipMalloc(&(handle->infoLU_array), nbytes), ROCSOLVER_STATUS_ALLOC_FAILED);
            for(int ibatch = 0; ibatch < batch_count; ibatch++)
            {
                handle->infoLU_array[ibatch] = 0;
            };
        };

        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            csrilu02Info_t infoLU = handle->infoLU_array[ibatch];
            if(infoLU == nullptr)
            {
                HIPSPARSE_CHECK(hipsparseCreateCsrilu02Info(&infoLU),
                                ROCSOLVER_STATUS_INTERNAL_ERROR);
            };
            handle->infoLU_array[ibatch] = infoLU;
        };
    };

    {
        size_t const nbytes = sizeof(int) * (n + 1);
        HIP_CHECK(hipMalloc((void**)&csrRowPtrLU, nbytes), ROCSOLVER_STATUS_ALLOC_FAILED);
    };

    if(csrRowPtrLU == nullptr)
    {
        return (ROCSOLVER_STATUS_ALLOC_FAILED);
    };

    //  ------------------
    //  Perform LU = L + U
    //  ------------------
    {
        int nrow = n;
        int ncol = n;

        double alpha = 1;
        double beta = 1;

        HIPSPARSE_CHECK(hipsparseSetPointerMode(hipsparse_handle, HIPSPARSE_POINTER_MODE_HOST),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        // ------------
        // setup buffer
        // ------------
        void* pBuffer = nullptr;
        size_t bufferSizeInBytes = 1;

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

        // ------------------------------
        // estimate number of non-zeros
        // ------------------------------
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

        // -----------------------------------
        // allocate storage for csrValLU_array
        // -----------------------------------
        {
            HIP_CHECK(hipMalloc((void**)&(handle->csrValLU_array), sizeof(double*) * batch_count),
                      ROCSOLVER_STATUS_ALLOC_FAILED);
            if(handle->csrValLU_array == nullptr)
            {
                return (ROCSOLVER_STATUS_ALLOC_FAILED);
            };

            for(int ibatch = 0; ibatch < batch_count; ibatch++)
            {
                double* csrValLU = nullptr;
                size_t const nbytes = sizeof(double) * nnz_LU;
                HIP_CHECK(hipMalloc((void**)&csrValLU, nbytes), ROCSOLVER_STATUS_ALLOC_FAILED);
                if(csrValLU == nullptr)
                {
                    return (ROCSOLVER_STATUS_ALLOC_FAILED);
                };
                handle->csrValLU_array[ibatch] = csrValLU;
            };
        };

        handle->n = n;
        handle->nnz_LU = nnz_LU;
        // -------------------------------------------
        // Perform sparse matrix addition, LU = L + U
        // -------------------------------------------

        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            double* csrValLU = handle->csrValLU_array[ibatch];

            HIPSPARSE_CHECK(hipsparseDcsrgeam2(hipsparse_handle, nrow, ncol, &alpha, descrL, nnzL,
                                               csrValL, csrRowPtrL, csrColIndL, &beta, descrU, nnzU,
                                               csrValU, csrRowPtrU, csrColIndU, descrLU, csrValLU,
                                               csrRowPtrLU, csrColIndLU, pBuffer),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
        };

        HIP_CHECK(hipFree(pBuffer), ROCSOLVER_STATUS_INTERNAL_ERROR);
        pBuffer = nullptr;

        handle->descrL = descrL;
        handle->descrU = descrU;

        handle->descrLU = descrLU;
        // handle->csrValLU = csrValLU;
        handle->csrRowPtrLU = csrRowPtrLU;
        handle->csrColIndLU = csrColIndLU;

        handle->infoL = infoL;
        handle->infoU = infoU;
    };

    // ---------------------------------
    // setup row and column permutations
    // ---------------------------------

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
        // -------------------------
        // Create inverse permutation Q_old2new[]
        //
        // inew = Q_new2old[ iold ]
        // Q_old2new[ iold ] = inew;
        // -------------------------
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

    // -----------------------------
    // copy the values of A into L+U
    // -----------------------------

    istat = rocsolverRfBatchResetValues(batch_count, n, nnzA, csrRowPtrA, csrColIndA, csrValA_array,
                                        P, Q, handle);

    // clean up

    {
        bool const need_cleanup_csrValA_array = (csrValA_array != csrValA_array_in);
        if(need_cleanup_csrValA_array)
        {
            for(int ibatch = 0; ibatch < batch_count; ibatch++)
            {
                HIP_CHECK(hipFree(csrValA_array[ibatch]), ROCSOLVER_STATUS_INTERNAL_ERROR);
                csrValA_array[ibatch] = nullptr;
            };
            HIP_CHECK(hipFree(csrValA_array), ROCSOLVER_STATUS_INTERNAL_ERROR);
            csrValA_array = nullptr;
        };
    };

    return (istat);
}
