

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
#include "geblt_common.h"

#pragma once
#ifndef ROCSOLVER_ADJUST_BATCH_HPP
#define ROCSOLVER_ADJUST_BATCH_HPP

template <typename T, typename I>
GLOBAL_FUNCTION void
    rocsolver_adjust_batch_kernel(bool const is_add, T* A_array[], I const offset, I const batch_count)
{
#ifdef USE_GPU
    I const ibatch_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const ibatch_inc = blockDim.x * gridDim.x;
#else
    I const ibatch_start = 0;
    I const ibatch_inc = 1;
#endif
    for(I ibatch = ibatch_start; ibatch < batch_count; ibatch += ibatch_inc)
    {
        if(is_add)
        {
            A_array[ibatch] += offset;
        }
        else
        {
            A_array[ibatch] -= offset;
        };
    };
    return;
};

template <typename T, typename I>
rocblas_status rocsolver_adjust_batch(rocblas_handle handle,
                                      bool const is_add,
                                      T* A_array[],
                                      I const offset,
                                      I const batch_count)
{
    auto const nthreads = 4 * 32;
    auto const nblocks = (batch_count + (nthreads - 1)) / nthreads;

    if(batch_count >= 1)
    {
        hipStream_t stream;
        rocblas_get_stream(handle, &stream);

        hipLaunchKernelGGL((rocsolver_adjust_batch_kernel<T, I>), dim3(nblocks), dim3(nthreads), 0,
                           stream, is_add, A_array, offset, batch_count);
    };
    return (rocblas_status_success);
};

#endif
