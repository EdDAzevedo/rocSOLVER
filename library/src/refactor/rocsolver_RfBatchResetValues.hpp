
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
#ifndef ROCSOLVER_RFBATCHRESETVALUES_HPP
#define ROCSOLVER_RFBATCHRESETVALUES_HPP

#include "rocsolverRf.h"

#include "hip_check.h"
#include "hipsparse_check.h"
#include <assert.h>

#include "rocsolver_aXpbY.hpp"
#include "rocsolver_add_PAQ.hpp"

template <typename Iint, typename Ilong, typename T>
rocsolverStatus_t rocsolver_RfBatchResetValues_template(Iint batch_count,
                                                        Iint n,
                                                        Iint nnzA,
                                                        Ilong* csrRowPtrA,
                                                        Iint* csrColIndA,
                                                        T* csrValA_array[],
                                                        Iint* P,
                                                        Iint* Q,

                                                        rocsolverRfHandle_t handle)
{
    // ------------
    // Check arguments
    // ------------

    {
        bool isok_handle = (handle != nullptr) && (handle->hipsparse_handle != nullptr);
        if(!isok_handle)
        {
            return (ROCSOLVER_STATUS_NOT_INITIALIZED);
        };
    };

    {
        bool const isok_scalar = (batch_count >= 0) && (n >= 0) && (nnzA >= 0);

        if(!isok_scalar)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    {
        bool const isok
            = (csrRowPtrA != nullptr) && (csrColIndA != nullptr) && (csrValA_array != nullptr);

        if(!isok)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    {
        bool const isok_assume_unchanged = (batch_count == handle->batch_count) && (n == handle->n)
            && (nnzA == handle->nnzA) && (csrRowPtrA == handle->csrRowPtrA)
            && (csrColIndA == handle->csrColIndA);
        if(!isok_assume_unchanged)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    {
        // ---------------------------------
        // check pointers in csrValA_array[]
        // ---------------------------------
        int nerrors = 0;
        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            T const* const csrValA = csrValA_array[ibatch];
            bool const is_error = (csrValA == nullptr);
            if(is_error)
            {
                nerrors++;
            };
        };
        bool const isok_csrValA = (nerrors == 0);
        if(!isok_csrValA)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    int const* const P_new2old = handle->P_new2old;
    int const* const Q_new2old = handle->Q_new2old;
    int const* const Q_old2new = handle->Q_old2new;

    {
        bool const is_ok = (P == P_new2old) && (Q == Q_new2old);
        if(!is_ok)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    int const ibatch = 0;

    hipStream_t stream;
    HIPSPARSE_CHECK(hipsparseGetStream(handle->hipsparse_handle, &stream),
                    ROCSOLVER_STATUS_EXECUTION_FAILED);

    bool const no_reordering = (P == nullptr) && (Q == nullptr);
    if(no_reordering)
    {
        // ------------------------------------------
        // No row reordering and No column reordering
        // ------------------------------------------

        int const nrow = n;
        int const ncol = n;
        T const alpha = 1;
        T const beta = 0;
        int const* const Xp = csrRowPtrA;
        int const* const Xi = csrColIndA;

        int const* const Yp = handle->csrRowPtrLU;
        int const* const Yi = handle->csrColIndLU;

        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            T const* const Xx = csrValA_array[ibatch];
            T* const Yx = handle->csrValLU_array[ibatch];
            rocsolver_aXpbY_template<Iint, Ilong, T>(stream, nrow, ncol, alpha, Xp, Xi, Xx, beta,
                                                     Yp, Yi, Yx);
        };
    }
    else
    {
        // -----------------------------------------
        // need to perform row and column reordering
        // -----------------------------------------
        int const nrow = n;
        int const ncol = n;

        int const* const Ap = csrRowPtrA;
        int const* const Ai = csrColIndA;

        int const* const LUp = handle->csrRowPtrLU;
        int const* const LUi = handle->csrColIndLU;

        T const alpha = 1;
        T const beta = 0;

        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            T* const LUx = handle->csrValLU_array[ibatch];
            T const* const Ax = csrValA_array[ibatch];
            rocsolver_add_PAQ<Iint, Ilong, T>(stream, nrow, ncol, P_new2old, Q_old2new, alpha, Ap,
                                              Ai, Ax, beta, LUp, LUi, LUx);
        };
    };

    return (ROCSOLVER_STATUS_SUCCESS);
}

#endif
