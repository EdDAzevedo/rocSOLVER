
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
#ifndef RF_APPLYRS_HPP
#define RF_APPLYRS_HPP

#include "hip_check.h"
#include "hipsparse_check.h"
#include <hipsparse/hipsparse.h>

#ifndef BLOCKSIZE
#define BLOCKSIZE 1024
#endif

template <typename T>
static __global__ void rf_applyRs_kernel(int const n, T const* const Rs, T* const b)
{
    if((n <= 0) || (Rs == NULL))
    {
        return;
    };

    unsigned int i_start = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int i_inc = blockDim.x * gridDim.x;

    for(unsigned int i = i_start; i < n; i += i_inc)
    {
        if(Rs[i] != 0)
        {
            b[i] = b[i] / Rs[i];
        };
    };
}

template <typename T>
void rf_applyRs_template(hipStream_t streamId, int const n, T const* const d_Rs, T* const d_b)
{
    assert(d_b != NULL);
    unsigned int nthreads = BLOCKSIZE;
    unsigned int min_nblocks = (n + (nthreads - 1)) / nthreads;
    unsigned int nblocks = (min_nblocks <= 0) ? 1 : min_nblocks;

    rf_applyRs_kernel<<<dim3(nblocks), dim3(nthreads), 0, streamId>>>(n, d_Rs, d_b);
}

#endif
