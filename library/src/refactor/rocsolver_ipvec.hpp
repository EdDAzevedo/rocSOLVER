
/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSOLVER_IPVEC_HPP
#define ROCSOLVER_IPVEC_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#ifndef IPVEC_BLOCKSIZE
#define IPVEC_BLOCKSIZE 256
#endif

template <typename I>
__global__
    __launch_bounds__(IPVEC_BLOCKSIZE) void rocsolver_ipvec_kernel(I const n,
                                                                   I const* const Q_new2old, /* input */
                                                                   I* const Q_old2new /* output */
    )
{
    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;

    for(I i = i_start; i < n; i += i_inc)
    {
        I const inew = i;
        I const iold = Q_new2old[inew];
        Q_old2new[iold] = inew;
    };
}

template <typename I>
void rocsolver_ipvec_template(hipStream_t stream, I const n, I const* const Q_new2old, I* const Q_old2new)
{
    if(n <= 0)
    {
        return;
    };
    if((Q_new2old == nullptr) || (Q_old2new == nullptr))
    {
        return;
    };

    int const nblocks = (n + (IPVEC_BLOCKSIZE - 1)) / IPVEC_BLOCKSIZE;
    rocsolver_ipvec_kernel<I>
        <<<dim3(nblocks), dim3(IPVEC_BLOCKSIZE), 0, stream>>>(n, Q_new2old, Q_old2new);
}

#endif
