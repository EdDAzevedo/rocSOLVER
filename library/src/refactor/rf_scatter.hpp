
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
#ifndef RF_SCATTER_HPP
#define RF_SCATTER_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <assert.h>

#ifndef SCATTER_THREADS
#define SCATTER_THREADS 1024
#endif

template <typename Iint, typename T>
static __global__  void rf_scatter_kernel(
                                        Iint const n, 
                                        Iint const * const P, 
                                        T    const * const src, 
                                        T          * const dest)
{


    Iint const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    Iint const i_inc = blockDim.x * gridDim.x;

    for(Iint i = i_start; i < n; i += i_inc)
    {
        Iint const ip = P[ i ];
        assert( (0 <= ip) && (ip < n) );

        dest[ ip  ] = src[ i ];
    };
}

template <typename Iint, typename T>
static void rf_scatter(hipStream_t streamId, 
                       Iint const n, 
                       Iint const * const P, 
                       T    const * const src, 
                       T          * const dest )
{
    int const nthreads = SCATTER_THREADS;
    int const max_nblocks = 32*1024;
    int const nblocks = min( max_nblocks, (n + (nthreads - 1)) / nthreads);

    rf_scatter_kernel<<<dim3(nblocks), dim3(nthreads), 0, streamId>>>(n, P, src, dest );
}

#endif

