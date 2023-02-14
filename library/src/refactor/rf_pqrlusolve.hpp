
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
#ifndef RF_PQRLUSOLVE_HPP
#define RF_PQRLUSOLVE_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>

#include "rf_common.hpp"

#include "rf_lusolve.hpp"

#include "rf_applyRs.hpp"
#include "rf_gather.hpp"
#include "rf_scatter.hpp"

template <typename Iint, typename Ilong, typename T>
static rocsolverStatus_t rf_pqrlusolve(rocsolverRfHandle_t handle,
                                       Iint const n,
                                       Iint* const P_new2old,
                                       Iint* const Q_new2old,
                                       T* const Rs,
                                       Ilong* const LUp,
                                       Iint* const LUi,
                                       T* const LUx, /* LUp,LUi,LUx  are in CSR format */
                                       T* const brhs,
                                       T* Temp)
{
    /*
    -------------------------------------------------
    Rs \ (P * A * Q) = LU
    solve A * x = b
       P A Q * (inv(Q) x) = P b
       { Rs \ (P A Q) } * (inv(Q) x) = Rs \ (P b)
       
       (LU) xhat = bhat,  xhat = inv(Q) x, or Q xhat = x,
                          bhat = Rs \ (P b)
    -------------------------------------------------
*/

    {
        bool const isok_arg
            = (LUp != nullptr) && (LUi != nullptr) && (LUx != nullptr) && (brhs != nullptr);
        if(!isok_arg)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;
    try
    {
        bool const need_apply_P = (P_new2old != nullptr);
        bool const need_apply_Q = (Q_new2old != nullptr);
        bool const need_apply_Rs = (Rs != nullptr);

        hipStream_t stream = handle->streamId.data();

        T* const d_brhs = brhs;
        T* const d_bhat = Temp;
        T* const d_Rs = Rs;

        if(need_apply_P)
        {
            // ------------------------------
            // bhat[k] = brhs[ P_new2old[k] ]
            // ------------------------------

            rf_gather(stream, n, P_new2old, d_brhs, d_bhat);
        }
        else
        {
            // -----------------
            // bhat[k] = brhs[k]
            // -----------------

            thrust::copy(d_brhs, d_brhs + n, d_bhat);
        };

        if(need_apply_Rs)
        {
            // -------------------------
            // bhat[k] = bhat[k] / Rs[k]
            // -------------------------
            rf_applyRs(stream, n, d_Rs, d_bhat);
        };

        // -----------------------------------------------
        // prepare to call triangular solvers rf_lusolve()
        // -----------------------------------------------

        {
            Ilong const nnz = LUp[n] - LUp[0];

            // ---------------------------------------
            // allocate device memory and copy LU data
            // ---------------------------------------

            Ilong* const d_LUp = LUp;
            Iint* const d_LUi = LUi;
            T* const d_LUx = LUx;
            T* const d_Temp = Temp;

            rocsolverStatus_t const istat_lusolve
                = rf_lusolve(handle, n, nnz, d_LUp, d_LUi, d_LUx, d_bhat, d_Temp);
            bool const isok_lusolve = (istat_lusolve == ROCSOLVER_STATUS_SUCCESS);
            if(!isok_lusolve)
            {
                throw std::runtime_error(__FILE__);
            };
        };

        if(need_apply_Q)
        {
            // -------------------------------
            // brhs[ Q_new2old[i] ] = bhat[i]
            // -------------------------------
            rf_scatter(stream, n, Q_new2old, d_bhat, d_brhs);
        }
        else
        {
            // ---------------------
            // brhs[ k ] = bhat[ k ]
            // ---------------------
            thrust::copy(d_bhat, d_bhat + n, d_brhs);
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

#endif
