
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
        bool const isok = (handle != nullptr);

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

template <typename Iint, typename Ilong, typename T>
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
    int const idebug = 1;
    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;

    try
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
        handle->nnzL = nnzL;
        handle->nnzU = nnzU;

        handle->nnzLU = (nnzL - n) + nnzU;

        {
            // -------------
            // copy matrix L
            // -------------
            handle->csrRowPtrL.resize(n + 1);
            handle->csrColIndL.resize(nnzL);
            handle->csrValL.resize(nnzL);

            thrust::copy(csrRowPtrL_in, csrRowPtrL_in + nnzL, handle->csrRowPtrL.begin());
            thrust::copy(csrColIndL_in, csrColIndL_in + nnzL, handle->csrColIndL.begin());
            thrust::copy(csrValL_in, csrValL_in + nnzL, handle->csrValL.begin());
        };

        {
            // -------------
            // copy matrix U
            // -------------
            handle->csrRowPtrU.resize(n + 1);
            handle->csrColIndU.resize(nnzU);
            handle->csrValU.resize(nnzU);

            thrust::copy(csrRowPtrU_in, csrRowPtrU_in + nnzU, handle->csrRowPtrU.begin());
            thrust::copy(csrColIndU_in, csrColIndU_in + nnzU, handle->csrColIndU.begin());
            thrust::copy(csrValU_in, csrValU_in + nnzU, handle->csrValU.begin());
        };

        {
            // ---------
            // copy P, Q
            // ---------
            handle->P_new2old.resize(n);
            handle->Q_new2old.resize(n);
            handle->Q_old2new.resize(n);

            thrust::copy(P_in, P_in + n, handle->P_new2old.begin());
            thrust::copy(Q_in, Q_in + n, handle->Q_new2old.begin());

            // -------------------------
            // generate inverse permutation Q_old2new[]
            //
            // inew = Q_new2old[ iold ]
            // Q_old2new[ iold ] = inew;
            // -------------------------
            rocsolver_ipvec_template(handle->streamId.data(), n, handle->Q_new2old.data().get(),
                                     handle->Q_old2new.data().get());
        };

        if(idebug >= 1)
        {
            printf("%s : %d\n", __FILE__, __LINE__);
        };

        // --------------------------------
        // setup storage for csrValLU_array
        // --------------------------------

        size_t const nnzLU = (nnzL - n) + nnzU;
        handle->nnzLU = nnzLU;

        handle->csrRowPtrLU.resize(n + 1);
        handle->csrColIndLU.resize(nnzLU);

        size_t const ialign = handle->ialign;
        size_t isize = ((nnzLU + (ialign - 1)) / ialign) * ialign;

        handle->csrValLU_array.resize(isize * batch_count);

        T const zero = 0;
        thrust::fill(handle->csrValLU_array.begin(), handle->csrValLU_array.end(), zero);

        if(idebug >= 1)
        {
            printf("%s : %d\n", __FILE__, __LINE__);
        };

        // -------------
        // setup descrL
        // -------------
        {
            THROW_IF_HIPSPARSE_ERROR(
                hipsparseSetMatType(handle->descrL.data(), HIPSPARSE_MATRIX_TYPE_TRIANGULAR));

            THROW_IF_HIPSPARSE_ERROR(
                hipsparseSetMatIndexBase(handle->descrL.data(), HIPSPARSE_INDEX_BASE_ZERO));

            THROW_IF_HIPSPARSE_ERROR(
                hipsparseSetMatFillMode(handle->descrL.data(), HIPSPARSE_FILL_MODE_LOWER));

            THROW_IF_HIPSPARSE_ERROR(
                hipsparseSetMatDiagType(handle->descrL.data(), HIPSPARSE_DIAG_TYPE_UNIT));
        };

        {
            // -------------
            // setup descrU
            // -------------
            THROW_IF_HIPSPARSE_ERROR(
                hipsparseSetMatType(handle->descrU.data(), HIPSPARSE_MATRIX_TYPE_TRIANGULAR));

            THROW_IF_HIPSPARSE_ERROR(
                hipsparseSetMatIndexBase(handle->descrU.data(), HIPSPARSE_INDEX_BASE_ZERO));

            THROW_IF_HIPSPARSE_ERROR(
                hipsparseSetMatFillMode(handle->descrU.data(), HIPSPARSE_FILL_MODE_UPPER));

            THROW_IF_HIPSPARSE_ERROR(
                hipsparseSetMatDiagType(handle->descrU.data(), HIPSPARSE_DIAG_TYPE_NON_UNIT));
        };

        {
            // -------------
            // setup descrLU
            // -------------
            THROW_IF_HIPSPARSE_ERROR(
                hipsparseSetMatType(handle->descrLU.data(), HIPSPARSE_MATRIX_TYPE_GENERAL));

            THROW_IF_HIPSPARSE_ERROR(
                hipsparseSetMatIndexBase(handle->descrLU.data(), HIPSPARSE_INDEX_BASE_ZERO));
        };

        //  ------------------
        //  Perform LU = L + U
        //  ------------------

        if(idebug >= 1)
        {
            printf("%s : %d\n", __FILE__, __LINE__);
        };

        for(Iint ibatch = 0; ibatch < batch_count; ibatch++)
        {
            Iint const nrow = n;
            Iint const ncol = n;

            Ilong const* const Lp = handle->csrRowPtrL.data().get();
            Iint const* const Li = handle->csrColIndL.data().get();
            T const* const Lx = handle->csrValL.data().get();

            Ilong const* const Up = handle->csrRowPtrU.data().get();
            Iint const* const Ui = handle->csrColIndU.data().get();
            T const* const Ux = handle->csrValU.data().get();

            Ilong* const LUp = handle->csrRowPtrLU.data().get();
            Iint* const LUi = handle->csrColIndLU.data().get();

            size_t const ialign = handle->ialign;
            size_t const isize = ((nnzLU + (ialign - 1)) / ialign) * ialign;
            size_t const offset = ibatch * isize;
            T* const LUx = handle->csrValLU_array.data().get() + offset;

            hipStream_t const stream = handle->streamId.data();
            rocsolverStatus_t istat
                = rf_sumLU(stream, nrow, ncol, Lp, Li, Lx, Up, Ui, Ux, LUp, LUi, LUx);
            bool const isok = (istat == ROCSOLVER_STATUS_SUCCESS);
            if(!isok)
            {
                throw std::runtime_error(__FILE__);
            };
        };

        if(idebug >= 1)
        {
            printf("%s : %d\n", __FILE__, __LINE__);
        };

        // -----------------------------
        // copy the values of A into L+U
        // -----------------------------

        rocsolverStatus_t const istat_ResetValues
            = rocsolverRfBatchResetValues(batch_count, n, nnzA, csrRowPtrA_in, csrColIndA_in,
                                          csrValA_array_in, P_in, Q_in, handle);
        bool const isok_ResetValues = (istat_ResetValues == ROCSOLVER_STATUS_SUCCESS);
        if(!isok_ResetValues)
        {
            throw std::runtime_error(__FILE__);
        };

        // ----------------
        // allocate buffer
        // ----------------

        {
            hipsparseOperation_t transL = HIPSPARSE_OPERATION_NON_TRANSPOSE;
            hipsparseOperation_t transU = HIPSPARSE_OPERATION_NON_TRANSPOSE;

            size_t bufferSize = sizeof(double);

            int stmp = 0;

            {
                int* Li = handle->csrColIndL.data().get();
                int* Lp = handle->csrRowPtrL.data().get();
                T* Lx = handle->csrValL.data().get();

                THROW_IF_HIPSPARSE_ERROR(
                    hipsparseDcsrsv2_bufferSize(handle, transL, n, nnzL, handle->descrL.data(), Lx,
                                                Lp, Li, handle->infoL.data(), &stmp));
            };

            if(stmp > bufferSize)
            {
                bufferSize = stmp;
            }

            stmp = 0;

            {
                int* Ui = handle->csrColIndU.data().get();
                int* Up = handle->csrRowPtrU.data().get();
                T* Ux = handle->csrValU.data().get();

                THROW_IF_HIPSPARSE_ERROR(
                    hipsparseDcsrsv2_bufferSize(handle, transU, n, nnzU, handle->descrU.data(), Ux,
                                                Up, Ui, handle->infoU.data(), &stmp));
            };

            if(stmp > bufferSize)
            {
                bufferSize = stmp;
            }

            handle->infoLU_array.resize(batch_count);
            for(int ibatch = 0; ibatch < batch_count; ibatch++)
            {
                int isize = 1;
                int const n = handle->n;
                int const nnz = handle->nnzLU;

                // ----------------------------
                // Here: note double precision
                // TODO: more precision types
                // ----------------------------
                Ilong const* const LUp = handle->csrRowPtrLU.data().get();
                Iint const* const LUi = handle->csrColIndLU.data().get();

                size_t ialign = handle->ialign;
                size_t const mat_size = ((nnzLU + (ialign - 1)) / ialign) * ialign;
                size_t const offset = ibatch * mat_size;

                double* LUx = handle->csrValLU_array.data().get() + offset;

                hipsparseHandle_t const hipsparse_handle = handle->hipsparse_handle.data();
                hipsparseMatDescr_t const descrLU = handle->descrLU.data();
                csrilu02Info_t infoLU = (handle->infoLU_array[ibatch]).data();

                THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrilu02_bufferSize(
                    hipsparse_handle, n, nnz, descrLU, LUx, LUp, LUi, infoLU, &isize));

                stmp = max(isize, stmp);
            };

            if(stmp > bufferSize)
            {
                bufferSize = stmp;
            }

            handle->buffer.resize(bufferSize);
        };
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
}
