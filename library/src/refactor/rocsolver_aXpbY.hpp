
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
#ifndef ROCSOLVER_AXPBY_HPP
#define ROCSOLVER_AXPBY_HPP

#ifndef AXPBY_MAX_THDS
#define AXPBY_MAX_THDS 256
#endif

template <typename Iint, typename Ilong, typename T>
static __global__ __launch_bounds__(AXPBY_MAX_THDS) void rocsolver_aXpbY_kernel(Iint const nrow,
                                                                         Iint const ncol,
                                                                         T const alpha,
                                                                         Iint const* const Xp,
                                                                         Iint const* const Xi,
                                                                         T const* const Xx,
                                                                         T const beta,
                                                                         Iint const* const Yp,
                                                                         Iint const* const Yi,
                                                                         T* const Yx)
{
    /*
 ------------------------------------------------
 Perform  Y = alpha * X + beta * Y
 where sparsity pattern of matrix X is a subset of
 sparsity pattern of matrix Y
 ------------------------------------------------
*/
    {
        bool const isok = (nrow >= 1) && (ncol >= 1) && (Xp != NULL) && (Xi != NULL) && (Xx != NULL)
            && (Yp != NULL) && (Yi != NULL) && (Yx != NULL);
        if(!isok)
        {
            return;
        };
    }

#include "rf_search.hpp"

    Iint const irow_start = threadIdx.x + blockIdx.x * blockDim.x;
    Iint const irow_inc = blockDim.x * gridDim.x;

    for(Iint irow = irow_start; irow < nrow; irow += irow_inc)
    {
        Ilong const kx_start = Xp[irow];
        Ilong const kx_end = Xp[irow + 1];
        Ilong const ky_start = Yp[irow];
        Ilong const ky_end = Yp[irow + 1];

        if(beta == 0)
        {
            for(Ilong ky = ky_start; ky < ky_end; ky++)
            {
                Yx[ky] = 0;
            };
        };

        if(alpha == 0)
        {
            /*
         -------------------
         just scale matrix Y
         -------------------
         */
            for(Ilong ky = ky_start; ky < ky_end; ky++)
            {
                Yx[ky] *= beta;
            };
        }
        else
        {
            for(Ilong kx = kx_start; kx < kx_end; kx++)
            {
                Iint const jcol = Xi[kx];

                /*
        ---------------------
        perform search
        ---------------------
        */
                bool is_found = false;
                {
                    Iint const key = jcol;
                    Iint const len = (ky_end - ky_start);
                    Iint const* const arr = &(Yp[ky_start]);

                    Iint const ipos = rf_search(len, arr, key);
                    is_found = (0 <= ipos) && (ipos < len) && (arr[ipos] == key);

                    if(is_found)
                    {
                        Ilong const ky = ky_start + ipos;
                        Yx[ky] = alpha * Xx[kx] + beta * Yx[ky];
                    };
                };
            };
        };
    };
}

template <typename Iint, typename Ilong, typename T>
void rocsolver_aXpbY_template(hipStream_t stream,

                              Iint const nrow,
                              Iint const ncol,
                              T const alpha,
                              Iint const* const Xp,
                              Iint const* const Xi,
                              T const* const Xx,
                              T const beta,
                              Iint const* const Yp,
                              Iint const* const Yi,
                              T* const Yx)
{
    Iint const nthreads = AXPBY_MAX_THDS;
    Iint const nblocks = (nrow + (nthreads - 1)) / nthreads;

    rocsolver_aXpbY_kernel<Iint, Ilong, T><<<dim3(nblocks), dim3(nthreads), 0, stream>>>(
        nrow, ncol, alpha, Xp, Xi, Xx, beta, Yp, Yi, Yx);
}

#endif
