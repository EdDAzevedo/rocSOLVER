
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
#pragma once
#include "rf_common.hpp"

#include "rf_sumLU.hpp"
#include "rocsolver_ipvec.hpp"

template <typename Iint, typename Ilong, typename T>
rocsolverStatus_t rocsolver_RfBatchSetup_checkargs(Iint batch_count,
                                                   Iint n,

                                                   Ilong nnzA,
                                                   Ilong* csrRowPtrA,
                                                   Iint* csrColIndA,
                                                   T* csrValA_array[],

                                                   Ilong nnzL,
                                                   Ilong* csrRowPtrL,
                                                   Iint* csrColIndL,
                                                   T* csrValL,

                                                   Ilong nnzU,
                                                   Ilong* csrRowPtrU,
                                                   Iint* csrColIndU,
                                                   T* csrValU,

                                                   Iint* P,
                                                   Iint* Q,
                                                   rocsolverRfHandle_t handle)
{
    // ---------------
    // check arguments
    // ---------------
    {
        bool const isok = (handle != nullptr) && (handle->hipsparse_handle != nullptr);

        if(!isok)
        {
            return (ROCSOLVER_STATUS_NOT_INITIALIZED);
        };
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

    {
        for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
        {
            if(csrValA_array[ibatch] == nullptr)
            {
                return (ROCSOLVER_STATUS_INVALID_VALUE);
            };
        };
    };

    return (ROCSOLVER_STATUS_SUCCESS);
};

template <bool MAKE_COPY, typename Iint, typename Ilong, typename T>
rocsolverStatus_t rocsolverRfBatchSetupDevice_impl(/* Input (in the device memory) */
                                                   Iint batch_count,
                                                   Iint n,

                                                   Ilong nnzA,
                                                   Ilong* csrRowPtrA_in,
                                                   Iint* csrColIndA_in,
                                                   T** csrValA_array_in,

                                                   Ilong nnzL,
                                                   Ilong* csrRowPtrL_in,
                                                   Iint* csrColIndL_in,
                                                   T* csrValL_in,

                                                   Ilong nnzU,
                                                   Ilong* csrRowPtrU_in,
                                                   Iint* csrColIndU_in,
                                                   T* csrValU_in,

                                                   Iint* P_in,
                                                   Iint* Q_in,

                                                   /* Output */
                                                   rocsolverRfHandle_t handle)
{
    // check args
    {
        rocsolverStatus_t istat = rocsolver_RfBatchSetup_checkargs(
            batch_count, n, nnzA, csrRowPtrA_in, csrColIndA_in, csrValA_array_in, nnzL,
            csrRowPtrL_in, csrColIndL_in, csrValL_in, nnzU, csrRowPtrU_in, csrColIndU_in,
            csrValU_in, P_in, Q_in, handle);
        if(istat != ROCSOLVER_STATUS_SUCCESS)
        {
            return (istat);
        };
    };

    handle->n = n;
    handle->batch_count = batch_count;
    handle->nnzA = nnzA;
    handle->nnzL = nnzL;
    handle->nnzU = nnzU;

    hipsparseHandle_t const hipsparse_handle = handle->hipsparse_handle;

    Ilong* csrRowPtrA = csrRowPtrA_in;
    Iint* csrColIndA = csrColIndA_in;
    T** csrValA_array = csrValA_array_in;

    Ilong* csrRowPtrL = csrRowPtrL_in;
    Iint* csrColIndL = csrColIndL_in;
    T* csrValL = csrValL_in;

    Ilong* csrRowPtrU = csrRowPtrU_in;
    Iint* csrColIndU = csrColIndU_in;
    T* csrValU = csrValU_in;

    Iint* P = P_in;
    Iint* Q = Q_in;

    {
        // --------------------------------
        // setup storage for csrValLU_array
        // --------------------------------
        handle->batch_count = batch_count;
        if(handle->csrValLU_array != nullptr)
        {
            for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
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

        HIP_CHECK(hipHostMalloc(&(handle->csrValLU_array), sizeof(T*) * batch_count,
                                hipHostMallocPortable),
                  ROCSOLVER_STATUS_ALLOC_FAILED);
        if(handle->csrValLU_array == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
        {
            T* csrValLU = nullptr;
            size_t const nbytes_L = sizeof(T) * nnzL;
            size_t const nbytes_U = sizeof(T) * nnzU;
            size_t const nbytes_LU = nbytes_L + nbytes_U - n;
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
        size_t const nbytes_PQ = sizeof(Iint) * n;

        HIP_CHECK(hipMalloc(&P, nbytes_PQ), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&Q, nbytes_PQ), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMemcpyDtoD(P, P_in, nbytes_PQ), ROCSOLVER_STATUS_EXECUTION_FAILED);

        HIP_CHECK(hipMemcpyDtoD(Q, Q_in, nbytes_PQ), ROCSOLVER_STATUS_EXECUTION_FAILED);

        // allocate and copy A
        size_t const nbytes_RowPtrA = sizeof(Ilong) * (n + 1);
        size_t const nbytes_ColIndA = sizeof(Iint) * nnzA;

        HIP_CHECK(hipMalloc(&csrRowPtrA, nbytes_RowPtrA), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&csrColIndA, nbytes_ColIndA), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMemcpyDtoD(csrRowPtrA, csrRowPtrA_in, nbytes_RowPtrA),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);
        HIP_CHECK(hipMemcpyDtoD(csrColIndA, csrColIndA_in, nbytes_ColIndA),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);

        // -------------------------------
        // allocate and copy csrValA_array
        // -------------------------------
        {
            size_t const nbytes_csrValA_array = sizeof(T*) * batch_count;
            HIP_CHECK(hipMalloc(&csrValA_array, nbytes_csrValA_array), ROCSOLVER_STATUS_ALLOC_FAILED);

            for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
            {
                T* csrValA = nullptr;
                size_t const nbytes_ValA = nnzA * sizeof(T);

                HIP_CHECK(hipMalloc(&csrValA, nbytes_ValA), ROCSOLVER_STATUS_ALLOC_FAILED);
                csrValA_array[ibatch] = csrValA;

                T* csrValA_in = csrValA_array_in[ibatch];
                HIP_CHECK(hipMemcpyDtoD(csrValA, csrValA_in, nbytes_ValA),
                          ROCSOLVER_STATUS_EXECUTION_FAILED);
            };
        };
        // -------------------
        // allocate and copy L
        // -------------------
        size_t const nbytes_RowPtrL = sizeof(Ilong) * (n + 1);
        size_t const nbytes_ColIndL = sizeof(Iint) * nnzL;
        size_t const nbytes_ValL = sizeof(T) * nnzL;

        HIP_CHECK(hipMalloc(&csrRowPtrL, nbytes_RowPtrL), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&csrColIndL, nbytes_ColIndL), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&csrValL, nbytes_ValL), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMemcpyDtoD(csrRowPtrL, csrRowPtrL_in, nbytes_RowPtrL),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);

        HIP_CHECK(hipMemcpyDtoD(csrColIndL, csrColIndL_in, nbytes_ColIndL),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);

        HIP_CHECK(hipMemcpyDtoD(csrValL, csrValL_in, nbytes_ValL), ROCSOLVER_STATUS_EXECUTION_FAILED);

        // -------------------
        // allocate and copy U
        // -------------------
        size_t const nbytes_RowPtrU = sizeof(Ilong) * (n + 1);
        size_t const nbytes_ColIndU = sizeof(Iint) * nnzU;
        size_t const nbytes_ValU = sizeof(T) * nnzU;

        HIP_CHECK(hipMalloc(&csrRowPtrU, nbytes_RowPtrU), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&csrColIndU, nbytes_ColIndU), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMalloc(&csrValU, nbytes_ValU), ROCSOLVER_STATUS_ALLOC_FAILED);

        HIP_CHECK(hipMemcpyDtoD(csrRowPtrU, csrRowPtrU_in, nbytes_RowPtrU),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);

        HIP_CHECK(hipMemcpyDtoD(csrColIndU, csrColIndU_in, nbytes_ColIndU),
                  ROCSOLVER_STATUS_EXECUTION_FAILED);

        HIP_CHECK(hipMemcpyDtoD(csrValU, csrValU_in, nbytes_ValU), ROCSOLVER_STATUS_EXECUTION_FAILED);
    };

    Iint* P_new2old = P;
    Iint* Q_new2old = Q;

    // ---------------------------------
    // form sparsity pattern for (L + U)
    // ---------------------------------

    Ilong* csrRowPtrLU = handle->csrRowPtrLU;
    Iint* csrColIndLU = handle->csrColIndLU;

    Iint const ibatch = 0;
    T* csrValLU = handle->csrValLU_array[ibatch];

    hipsparseMatDescr_t descrLU = handle->descrLU;
    Ilong nnzLU = 0;

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

            HIPSPARSE_CHECK(hipsparseSetMatFillMode(descrL, HIPSPARSE_FILL_MODE_LOWER),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);

            HIPSPARSE_CHECK(hipsparseSetMatDiagType(descrL, HIPSPARSE_DIAG_TYPE_UNIT),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);

            handle->descrL = descrL;
        };

        if(infoL == nullptr)
        {
            HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&infoL), ROCSOLVER_STATUS_INTERNAL_ERROR);
            handle->infoL = infoL;
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

            HIPSPARSE_CHECK(hipsparseSetMatFillMode(descrU, HIPSPARSE_FILL_MODE_UPPER),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);

            HIPSPARSE_CHECK(hipsparseSetMatDiagType(descrU, HIPSPARSE_DIAG_TYPE_NON_UNIT),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);

            handle->descrU = descrU;
        };

        if(infoU == nullptr)
        {
            HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&infoU), ROCSOLVER_STATUS_INTERNAL_ERROR);

            handle->infoU = infoU;
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

            handle->descrLU = descrLU;
        };

        if(handle->infoLU_array == nullptr)
        {
            size_t const nbytes = batch_count * sizeof(csrilu02Info_t);
            HIP_CHECK(hipMalloc(&(handle->infoLU_array), nbytes), ROCSOLVER_STATUS_ALLOC_FAILED);
            for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
            {
                handle->infoLU_array[ibatch] = 0;
            };
        };

        for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
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
        size_t const nbytes = sizeof(Ilong) * (n + 1);
        HIP_CHECK(hipMalloc((void**)&csrRowPtrLU, nbytes), ROCSOLVER_STATUS_ALLOC_FAILED);
        handle->csrRowPtrLU = csrRowPtrLU;
    };

    if(csrRowPtrLU == nullptr)
    {
        return (ROCSOLVER_STATUS_ALLOC_FAILED);
    };

    //  ------------------
    //  Perform LU = L + U
    //  ------------------

    bool constexpr use_sumLU = true;
    if(use_sumLU)
    {
        Ilong nnzLU = handle->nnzL + handle->nnzU - handle->n;
        handle->nnzLU = nnzLU;

        if(handle->csrValLU_array == nullptr)
        {
            size_t const nbytes = sizeof(T*) * batch_count;
            HIP_CHECK(hipMalloc((void**)&(handle->csrValLU_array), nbytes),
                      ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        if(handle->csrColIndLU == nullptr)
        {
            HIP_CHECK(hipMalloc((void**)&(handle->csrColIndLU), sizeof(Iint) * handle->nnzLU),
                      ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
        {
            if(handle->csrValLU_array[ibatch] == nullptr)
            {
                size_t const nbytes = sizeof(T) * handle->nnzLU;
                HIP_CHECK(hipMalloc((void**)&(handle->csrValLU_array[ibatch]), nbytes),
                          ROCSOLVER_STATUS_ALLOC_FAILED);
            };
        };

        for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
        {
            Iint const nrow = n;
            Iint const ncol = n;
            rocsolverStatus_t istat
                = rf_sumLU(handle, nrow, ncol, handle->csrRowPtrL, handle->csrColIndL,
                           handle->csrValL, handle->csrRowPtrU, handle->csrColIndU, handle->csrValU,

                           handle->csrRowPtrLU, handle->csrColIndLU, handle->csrValLU_array[ibatch]);
            if(istat != ROCSOLVER_STATUS_SUCCESS)
            {
                return (istat);
            };
        };
    }
    else
    {
        Iint const nrow = n;
        Iint const ncol = n;

        T const alpha = 1;
        T const beta = 1;

        HIPSPARSE_CHECK(hipsparseSetPointerMode(hipsparse_handle, HIPSPARSE_POINTER_MODE_HOST),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        // ------------
        // setup buffer
        // ------------
        void* pBuffer = nullptr;
        size_t bufferSizeInBytes = sizeof(T);

        // ------------------------------
        // hipsparseXcsrgeam2() computes
        // C = alpha * A + beta * B
        // ------------------------------

        HIPSPARSE_CHECK(hipsparseDcsrgeam2_bufferSizeExt(
                            hipsparse_handle, nrow, ncol, &alpha, descrL, nnzL, csrValL, csrRowPtrL,
                            csrColIndL, &beta, descrU, nnzU, csrValU, csrRowPtrU, csrColIndU,
                            descrLU, csrValLU, csrRowPtrLU, csrColIndLU, &bufferSizeInBytes),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        HIP_CHECK(hipMalloc(&pBuffer, bufferSizeInBytes), ROCSOLVER_STATUS_ALLOC_FAILED);

        // ------------------------------
        // estimate number of non-zeros
        // in L + U
        // ------------------------------
        HIPSPARSE_CHECK(hipsparseXcsrgeam2Nnz(hipsparse_handle, nrow, ncol, descrL, nnzL,
                                              csrRowPtrL, csrColIndL, descrU, nnzU, csrRowPtrU,
                                              csrColIndU, descrLU, csrRowPtrLU, &nnzLU, pBuffer),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        HIP_CHECK(hipMalloc((void**)&csrColIndLU, sizeof(Iint) * nnzLU),
                  ROCSOLVER_STATUS_ALLOC_FAILED);
        if(csrColIndLU == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        // -----------------------------------
        // allocate storage for csrValLU_array
        // -----------------------------------
        {
            HIP_CHECK(hipMalloc((void**)&(handle->csrValLU_array), sizeof(T*) * batch_count),
                      ROCSOLVER_STATUS_ALLOC_FAILED);
            if(handle->csrValLU_array == nullptr)
            {
                return (ROCSOLVER_STATUS_ALLOC_FAILED);
            };

            for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
            {
                T* csrValLU = nullptr;
                size_t const nbytes = sizeof(T) * nnzLU;
                HIP_CHECK(hipMalloc((void**)&csrValLU, nbytes), ROCSOLVER_STATUS_ALLOC_FAILED);
                if(csrValLU == nullptr)
                {
                    return (ROCSOLVER_STATUS_ALLOC_FAILED);
                };
                handle->csrValLU_array[ibatch] = csrValLU;
            };
        };

        handle->n = n;
        handle->nnzLU = nnzLU;
        // -------------------------------------------
        // Perform sparse matrix addition, LU = L + U
        // -------------------------------------------

        for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
        {
            T* csrValLU = handle->csrValLU_array[ibatch];

            HIPSPARSE_CHECK(hipsparseDcsrgeam2(hipsparse_handle, nrow, ncol, &alpha, descrL, nnzL,
                                               csrValL, csrRowPtrL, csrColIndL, &beta, descrU, nnzU,
                                               csrValU, csrRowPtrU, csrColIndU, descrLU, csrValLU,
                                               csrRowPtrLU, csrColIndLU, pBuffer),
                            ROCSOLVER_STATUS_INTERNAL_ERROR);
        };

        HIP_CHECK(hipFree(pBuffer), ROCSOLVER_STATUS_INTERNAL_ERROR);
        pBuffer = nullptr;
    };

    handle->csrRowPtrL = csrRowPtrL;
    handle->csrColIndL = csrColIndL;
    handle->csrValL = csrValL;

    handle->csrRowPtrU = csrRowPtrU;
    handle->csrColIndU = csrColIndU;
    handle->csrValU = csrValU;

    handle->csrRowPtrA = csrRowPtrA;
    handle->csrColIndA = csrColIndA;

    handle->descrL = descrL;
    handle->descrU = descrU;

    handle->descrLU = descrLU;
    // handle->csrValLU = csrValLU;
    handle->csrRowPtrLU = csrRowPtrLU;
    handle->csrColIndLU = csrColIndLU;

    handle->infoL = infoL;
    handle->infoU = infoU;

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

    hipStream_t streamId;
    HIPSPARSE_CHECK(hipsparseGetStream(handle->hipsparse_handle, &streamId),
                    ROCSOLVER_STATUS_INTERNAL_ERROR);
    {
        // -------------------------
        // Create inverse permutation Q_old2new[]
        //
        // inew = Q_new2old[ iold ]
        // Q_old2new[ iold ] = inew;
        // -------------------------
        Iint* Q_old2new = nullptr;
        HIP_CHECK(hipMalloc(&(Q_old2new), sizeof(Iint) * n), ROCSOLVER_STATUS_ALLOC_FAILED);
        if(Q_old2new == nullptr)
        {
            return (ROCSOLVER_STATUS_ALLOC_FAILED);
        };

        rocsolver_ipvec_template(streamId, n, Q_new2old, Q_old2new);

        handle->Q_old2new = Q_old2new;
    };

    // -----------------------------
    // copy the values of A into L+U
    // -----------------------------

    rocsolverStatus_t const istat = rocsolverRfBatchResetValues(
        batch_count, n, nnzA, csrRowPtrA, csrColIndA, csrValA_array, P, Q, handle);

    // ----------------
    // allocate buffer
    // ----------------

    {
        hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
        hipsparseOperation_t transL = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        hipsparseOperation_t transU = HIPSPARSE_OPERATION_NON_TRANSPOSE;

        size_t bufferSize = sizeof(double);

        int stmp = 0;
        HIPSPARSE_CHECK(hipsparseDcsrsv2_bufferSize(handle, transL, n, nnzL, descrL, csrValL,
                                                    csrRowPtrL, csrColIndL, infoL, &stmp),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        if(stmp > bufferSize)
        {
            bufferSize = stmp;
        }

        stmp = 0;
        HIPSPARSE_CHECK(hipsparseDcsrsv2_bufferSize(handle, transU, n, nnzU, descrU, csrValU,
                                                    csrRowPtrU, csrColIndU, infoU, &stmp),
                        ROCSOLVER_STATUS_INTERNAL_ERROR);

        if(stmp > bufferSize)
        {
            bufferSize = stmp;
        }

        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            int isize = 1;
            int const n = handle->n;
            int const nnz = handle->nnzLU;

            // ----------------------------
            // Here: note double precision
            // TODO: more precision types
            // ----------------------------
            HIPSPARSE_CHECK(hipsparseDcsrilu02_bufferSize(
                                handle->hipsparse_handle, n, nnz, handle->descrLU,
                                handle->csrValLU_array[ibatch], handle->csrRowPtrLU,
                                handle->csrColIndLU, handle->infoLU_array[ibatch], &isize),
                            ROCSOLVER_STATUS_EXECUTION_FAILED);
            stmp = max(isize, stmp);
        };

        if(stmp > bufferSize)
        {
            bufferSize = stmp;
        }

        if(handle->buffer != nullptr)
        {
            HIP_CHECK(hipFree(handle->buffer), ROCSOLVER_STATUS_INTERNAL_ERROR);
            handle->buffer = nullptr;
        };

        handle->buffer_size = bufferSize;
        HIP_CHECK(hipMalloc(&(handle->buffer), handle->buffer_size), ROCSOLVER_STATUS_ALLOC_FAILED);
    };

    // --------
    // clean up
    // --------

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
