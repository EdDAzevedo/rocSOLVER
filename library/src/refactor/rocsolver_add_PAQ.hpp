
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
#ifndef ROCSOLVER_ADD_PAQ_HPP
#define ROCSOLVER_ADD_PAQ_HPP

#include "rocsolver_refactor.h"
#include <assert.h>

#ifndef ADD_PAQ_MAX_THDS
#define ADD_PAQ_MAX_THDS 256
#endif

template <typename Iint, typename Ilong, typename T>
static __global__ __launch_bounds__(ADD_PAQ_MAX_THDS) void rocsolver_add_PAQ_kernel(Iint const nrow,
                                                                      Iint const ncol,
                                                                      Iint const* const P_new2old,
                                                                      Iint const* const Q_old2new,
                                                                      Iint const* const Ap,
                                                                      Iint const* const Ai,
                                                                      T const* const Ax,
                                                                      Iint const* const LUp,
                                                                      Iint const* const LUi,
                                                                      T* const LUx)
{
/*
 ------------------------
 inline lambda expression
 ------------------------
*/
#include "rf_search.hpp"

    /*
     -------------------------------------------
     If P_new2old, or Q_old2new is NULL, then treat as identity permutation
     -------------------------------------------
     */

    bool const has_P = (P_new2old != nullptr);
    bool const has_Q = (Q_old2new != nullptr);
    Iint const irow_start = threadIdx.x + blockIdx.x * blockDim.x;
    Iint const irow_inc = blockDim.x * gridDim.x;

    for(Iint irow = irow_start; irow < nrow; irow += irow_inc)
    {
        Ilong const kstart_LU = LUp[irow];
        Ilong const kend_LU = LUp[irow + 1];
        Iint const nz_LU = kend_LU - kstart_LU;

        /*
         -------------------
         initialize row to zeros
         -------------------
        */
        for(Iint k = 0; k < nz_LU; k++)
        {
            Ilong const k_lu = kstart_LU + k;
            LUx[k_lu] = 0;
        };

        Iint const irow_old = (has_P) ? P_new2old[irow] : irow;
        Ilong const kstart_A = Ap[irow_old];
        Ilong const kend_A = Ap[irow_old + 1];
        Iint const nz_A = kend_A - kstart_A;

        for(Iint k = 0; k < nz_A; k++)
        {
            Ilong const ka = kstart_A + k;

            Iint const jcol_old = Ai[ka];
            Iint const jcol = (has_Q) ? Q_old2new[jcol_old] : jcol_old;

            Iint const len = nz_LU;
            Iint ipos = len;
            {
                Iint const* const arr = &(LUi[kstart_LU]);
                Iint const key = jcol;

                ipos = rf_search(len, arr, key);
                bool const is_found = (0 <= ipos) && (ipos < len) && (arr[ipos] == key);
                assert(is_found);
            };

            Ilong const k_lu = kstart_LU + ipos;

            T const aij = Ax[ka];
            LUx[k_lu] += aij;
        };
    };
}

template <typename Iint, typename Ilong, typename T>
rocsolverStatus_t rocsolver_add_PAQ(hipStream_t stream,

                                    Iint const nrow,
                                    Iint const ncol,
                                    Iint const* const P_new2old,
                                    Iint const* const Q_old2new,
                                    Iint const* const Ap,
                                    Iint const* const Ai,
                                    T const* const Ax,
                                    Iint const* const LUp,
                                    Iint const* const LUi,
                                    T* const LUx)
{
    int const nthreads = ADD_PAQ_MAX_THDS;
    int const nblocks = (nrow + (nthreads - 1)) / nthreads;

    rocsolver_add_PAQ_kernel<Iint, Ilong, T><<<dim3(nthreads), dim3(nblocks), 0, stream>>>(
        nrow, ncol, P_new2old, Q_old2new, Ap, Ai, Ax, LUp, LUi, LUx);
    return (ROCSOLVER_STATUS_SUCCESS);
}
#endif
