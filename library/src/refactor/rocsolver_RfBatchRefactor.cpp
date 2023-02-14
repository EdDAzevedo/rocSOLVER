
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

#include "rf_common.hpp"

/*
---------------------------------------------
This routine performs the LU re-factorization
    A = L * U
uses the available parallelism on the GPU.  It is assumed that a prior
call to rocsolverRfAnalyze() was done in order to find the available
parallelism.

This routine may be called multiple times, once for each of the linear
systems:
   A_i  x_i = f_i

There are some constraints to the combination of algorithms used
for refactorization and solving routines, rocsolverRfBatchRefactor() and
rocsolverRfSolve().  The wrong combination generates the error code
ROCSOLVER_STATUS_INVALID_VALUE.
---------------------------------------------
 */

extern "C" {

rocsolverStatus_t rocsolverRfBatchRefactor(rocsolverRfHandle_t handle)
{
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;
    try
    {
        int const batch_count = handle->batch_count;

        int const n = handle->n;
        int const nnz = handle->nnzLU;
        hipsparseHandle_t const hipsparse_handle = handle->hipsparse_handle.data();

        hipsparseMatDescr_t const descrLU = handle->descrLU.data();
        int* const csrRowPtrLU = handle->csrRowPtrLU.data().get();
        int* const csrColIndLU = handle->csrColIndLU.data().get();

        //  ----------------------------------------------------
        //  numerical boost is disabled by default in cusolverRF
        //  ----------------------------------------------------
        double const effective_zero = handle->effective_zero;
        double const boost_val = handle->boost_val;

        // -------------------------------------------------------------
        // set enable_boost to 1 or disable by setting enable_boost to 0
        // -------------------------------------------------------------
        int const enable_boost = (boost_val > 0.0) ? 1 : 0;

        // ---------------------
        // perform factorization
        // ---------------------

        size_t const ialign = handle->ialign;
        size_t const isize = ((nnzLU + (ialign - 1)) / ialign) * ialign;
        void* const buffer = handle->buffer.data();

        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            size_t const offset = ibatch * isize;
            double* const csrValLU = handle->csrValLU_array.data().get() + offset;

            auto const infoLU = infoLU_array[ibatch].data();

            THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrilu02_numericBoost(
                hipsparse_handle, infoLU, enable_boost, &effective_zero, &boost_val));

            THROW_IF_HIPSPARSE_ERROR(hipsparseDcsrilu02(hipsparse_handle, n, nnz, descrLU, csrValLU,
                                                        csrRowPtrLU, csrColIndLU, infoLU, policy,
                                                        buffer));
        };

        // ----------------------------------------------------------------------
        // note, there is no checking for zero pivot
        // the user has to call "RfBatchZeroPivot()" to know which matrix failed
        // the LU factorization
        // ----------------------------------------------------------------------
    }
    catch(const std::bad_alloc& e)
    {
        istat_return = ROCSOLVER_STATUS_ALLOC_FAILED;
    }
    catch(const std::runtime_error& e)
    {
        istat_return = ROCSOLVER_STATUS_EXECUTION_ERROR;
    }
    catch(...)
    {
        istat_return = ROCSOLVER_STATUS_INTERNAL_ERROR;
    };

    return (istat_return);
};
};
