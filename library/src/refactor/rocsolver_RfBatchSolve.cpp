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
/*
 -------------------------------------------------------------
This routine performs the forward and backward solve with the upper
and lower triangular factors computed from the LU re-factorization
rocsolverRfRefactor() routine.

The routine can solve linear systems with multiple right-hand-sides (RHS):

  solve F = A X = (L U) X = L (U X) or   L Y = F, where Y = U X

This routine may be called multiple times, once for each of the linear
systems:

   A_i x_i = f_i

 -------------------------------------------------------------
*/

#include "rf_pqrlusolve.hpp"
#include "rocsolver_RfSolve.hpp"
#include <assert.h>

extern "C" {

rocsolverStatus_t rocsolverRfBatchSolve(
    /* Input (in the device memory) */
    rocsolverRfHandle_t handle,
    int* P,
    int* Q,
    int nrhs,
    double* Temp, /* dense matrix of size (ldt * nrhs), ldt >= n */
    int ldt,

    /* Input/Output (in the device memory) */

    // -----------------------------------------
    // dense matrix that contains right-hand side F
    // and solutions X of size (ldxf * nrhs)
    // -----------------------------------------
    double* XF_array[],

    /* Input */
    int ldxf)
{
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    int const batch_count = handle->batch_count;

    {
        bool const isok
            = (P != nullptr) && (Q != nullptr) && (Temp != nullptr) && (XF_array != nullptr);
        if(!isok)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };

        for(auto ibatch = 0; ibatch < batch_count; ibatch++)
        {
            if(XF_array[ibatch] == nullptr)
            {
                return (ROCSOLVER_STATUS_INVALID_VALUE);
            };
        };
    };

    double* const d_Temp = Temp;

    int const n = handle->n;
    if((n < 0) || (nrhs < 0))
    {
        return (ROCSOLVER_STATUS_INVALID_VALUE);
    };

    if((n == 0) || (nrhs == 0) || (batch_count == 0))
    {
        // no work
        return (ROCSOLVER_STATUS_SUCCESS);
    };

    {
        bool const isok_arguments = (XF_array != nullptr) && (ldxf >= n);
        if(!isok_arguments)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;
    try
    {
        int* const P_new2old = handle->P_new2old.data().get();
        int* const Q_new2old = handle->Q_new2old.data().get();

        int nerrors = 0;
        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            double* const XF = XF_array[ibatch];
            for(int irhs = 0; irhs < nrhs; irhs++)
            {
                double* const Rs = nullptr;
                double* const brhs = &(XF[ldxf * irhs]);
                int* const LUp = handle->csrRowPtrLU.data().get();
                int* const LUi = handle->csrColIndLU.data().get();

                int const nnzLU = handle->nnzLU;
                size_t const ialign = handle->ialign;
                size_t const isize = ((nnzLU + (ialign - 1)) / ialign) * ialign;
                size_t const offset = ibatch * isize;
                double* const LUx = handle->csrValLU_array.data().get() + offset;

                rocsolverStatus_t istat = rf_pqrlusolve(handle, n, nnzLU, P_new2old, Q_new2old, Rs,
                                                        LUp, LUi, LUx, brhs, d_Temp);

                bool const isok = (istat == ROCSOLVER_STATUS_SUCCESS);
                if(!isok)
                {
                    nerrors++;
                };
            };
        }; // end for ibatch

        RF_ASSERT(nerrors == 0);
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
};
};
