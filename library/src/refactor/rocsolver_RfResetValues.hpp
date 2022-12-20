
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
#ifndef ROCSOLVER_RFRESETVALUES_HPP
#define ROCSOLVER_RFRESETVALUES_HPP

#include "rocsolverRf.h"

#include "hip_check.h"
#include "hipsparse_check.h"
#include <assert.h>

#include "rocsolver_aXpbY.hpp"
#include "rocsolver_add_PAQ.hpp"

template <typename Iint, typename Ilong, typename T>
rocsolverStatus_t rocsolver_RfResetValues_template(Iint n,
                                                   Iint nnzA,
                                                   Iint* csrRowPtrA,
                                                   Iint* csrColIndA,
                                                   T* csrValA,
                                                   Iint* P,
                                                   Iint* Q,

                                                   rocsolverRfHandle_t handle)
{
    /*
   ------------
   Quick return
   ------------
   */

    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    bool const isok_scalar = (n >= 0) && (nnzA >= 0);
    if(!isok_scalar)
    {
        return (ROCSOLVER_STATUS_INVALID_VALUE);
    };

    bool const isok = (csrRowPtrA != nullptr) && (csrColIndA != nullptr) && (csrValA != nullptr);

    if(!isok)
    {
        return (ROCSOLVER_STATUS_INVALID_VALUE);
    };

    int const* const P_new2old = handle->P_new2old;
    int const* const Q_new2old = handle->Q_new2old;
    int const* const Q_old2new = handle->Q_old2new;

    bool const is_ok = (P == P_new2old) && (Q == Q_new2old);
    assert(is_ok);
    if(!is_ok)
    {
        return (ROCSOLVER_STATUS_INTERNAL_ERROR);
    };

    hipStream_t stream;
    HIPSPARSE_CHECK(hipsparseGetStream(handle->hipsparse_handle, &stream),
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    if((P == NULL) && (Q == NULL))
    {
        /*
    ------------------------------------------
    No row reordering and No column reordering
    ------------------------------------------
    */

        int const nrow = n;
        int const ncol = n;
        double const alpha = 1;
        double const beta = 0;
        int const* const Xp = csrRowPtrA;
        int const* const Xi = csrColIndA;
        double const* const Xx = csrValA;

        int const* const Yp = handle->csrRowPtrLU;
        int const* const Yi = handle->csrColIndLU;
        double* const Yx = handle->csrValLU;
        rocsolver_aXpbY_template<Iint, Ilong, T>(stream,

                                                 nrow, ncol, alpha, Xp, Xi, Xx, beta, Yp, Yi, Yx);
    }
    else
    {
        int const nrow = n;
        int const ncol = n;

        int const* const Ap = csrRowPtrA;
        int const* const Ai = csrColIndA;
        double const* const Ax = csrValA;

        int const* const LUp = handle->csrRowPtrLU;
        int const* const LUi = handle->csrColIndLU;
        double* const LUx = handle->csrValLU;

        rocsolver_add_PAQ<Iint, Ilong, T>(stream,

                                          nrow, ncol, P_new2old, Q_old2new, Ap, Ai, Ax, LUp, LUi,
                                          LUx);
    };

    return (ROCSOLVER_STATUS_SUCCESS);
}

#endif
