
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
        bool isok_handle = (handle != nullptr);
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

    int const* const P_new2old = handle->P_new2old.data().get();
    int const* const Q_new2old = handle->Q_new2old.data().get();
    int const* const Q_old2new = handle->Q_old2new.data().get();

    {
        bool const is_ok = (P == P_new2old) && (Q == Q_new2old) && (Q_old2new != nullptr);
        if(!is_ok)
        {
            return (ROCSOLVER_STATUS_INVALID_VALUE);
        };
    };

    {
        // -----------------------------------------
        // need to perform row and column reordering
        // -----------------------------------------
        int const nrow = n;
        int const ncol = n;

        int const* const Ap = csrRowPtrA;
        int const* const Ai = csrColIndA;

        int const* const LUp = handle->csrRowPtrLU.data().get();
        int const* const LUi = handle->csrColIndLU.data().get();

        T const alpha = 1;
        T const beta = 0;

        int const* const P_new2old = handle->P_new2old.data().get();
        int const* const Q_old2new = handle->Q_old2new.data().get();
        hipStream_t const stream = handle->streamId.data();

        size_t const ialign = handle->ialign;
        size_t const isize = ((nnzA + (ialign - 1)) / ialign) * ialign;

        handle->csrValLU_array.resize(batch_count * isize);

        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            size_t const offset = ibatch * isize;

            T* const LUx = handle->csrValLU_array.data().get() + offset;
            T const* const Ax = csrValA_array[ibatch];
            rocsolver_add_PAQ<Iint, Ilong, T>(stream, nrow, ncol, P_new2old, Q_old2new, alpha, Ap,
                                              Ai, Ax, beta, LUp, LUi, LUx);
        };
    };

    return (ROCSOLVER_STATUS_SUCCESS);
}

#endif
