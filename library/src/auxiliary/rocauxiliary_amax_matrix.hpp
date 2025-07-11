/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/
#pragma once

#if(0)
#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#endif

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include <cmath>
#include <complex>

ROCSOLVER_BEGIN_NAMESPACE

#ifndef HIP_CHECK
#define HIP_CHECK(fcn)               \
    {                                \
        auto const istat = (fcn);    \
        assert(istat == hipSuccess); \
    }
#endif

static int get_max_threads()
{
    return (1024);
}

static int get_num_cu(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMultiprocessorCount;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

template <typename I = int>
static I get_lds_size()
{
    I const default_lds_size = 64 * 1024;

    I lds_size = 0;
    I deviceId = 0;
    auto const istat_device = hipGetDevice(&deviceId);
    if(istat_device != hipSuccess)
    {
        return (default_lds_size);
    };
    auto const attr = hipDeviceAttributeMaxSharedMemoryPerBlock;
    auto istat_attr = hipDeviceGetAttribute(&lds_size, attr, deviceId);
    if(istat_attr != hipSuccess)
    {
        return (default_lds_size);
    };

    return (lds_size);
}

// -------------------------------------
// compute the max abs value of a matrix
// assume the p_amax(0:(batch_count-1)) array
// has been initialized to zero
// -------------------------------------
template <typename S, typename I, typename Istride, typename UA>
static __global__ void amax_matrix_kernel(I const nrows,
                                          I const ncols,
                                          S* const p_amax,

                                          UA A_,
                                          Istride const shift_A,
                                          I const ldA,
                                          Istride const stride_A,

                                          I const batch_count,
                                          void* work)
{
    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    I const tid = threadIdx.x + threadIdx.y * blockDim.x;
    I const nthreads = blockDim.x * blockDim.y;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const i_inc = blockDim.x * gridDim.x;

    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;
    I const j_inc = blockDim.y * gridDim.y;

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    extern __shared__ double ldsmem[];
    S* const sharedData = (S*)&(ldsmem[0]);

    auto idx2D = [](auto i, auto j, auto ld) { return (i + (j * static_cast<int64_t>(ld))); };

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        auto const Ap = load_ptr_batch(A_, bid, shift_A, stride_A);
        auto A = [=](auto i, auto j) { return (Ap[idx2D(i, j, ldA)]); };

        sharedData[tid] = 0;
        __syncthreads();

        S amax_thread = 0;
        for(auto j = j_start; j < ncols; j += j_inc)
        {
            for(auto i = i_start; i < nrows; i += i_inc)
            {
                auto const abs_aij = std::abs(A(i, j));
                amax_thread = (abs_aij > amax_thread) ? abs_aij : amax_thread;
            }
        }

        __syncthreads();

        sharedData[tid] = amax_thread;

        __synchthreads();

        // ---------------------------------
        // max reduction using shared memory
        // ---------------------------------
        for(int stride = nthreads / 2; stride > 0; stride /= 2)
        {
            if(tid < stride)
            {
                sharedData[tid] = std::max(sharedData[tid], sharedData[tid + stride]);
            }
            __syncthreads();
        }
        __syncthreads();

        if(tid == 0)
        {
            atomicMax(&(p_amax[bid]), sharedData[0]);
        }
    } // end for bid
}

// -----------------------------------
// compute the max abs entry of matrix
// -----------------------------------
template <typename S, typename I, typename Istride, typename UA>
static void amax_matrix(hipStream_t stream,
                        I const nrows,
                        I const ncols,
                        S* const p_amax,

                        UA A_,
                        Istride const shift_A,
                        I const ldA,
                        Istride const stride_A,

                        I const batch_count,
                        void* work)
{
    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    // ------------------------
    // initialize max value to zero
    // this is necessary for correctness
    // ------------------------
    {
        auto nbytes = sizeof(S) * batch_count;
        HIP_CHECK(hipMemsetAsync(&(p_amax[0]), 0, nbytes, stream));
    }

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    I const lds_size = get_lds_size();
    I const nx = 32;
    I const ny = 32;

    I const num_cu = get_num_cu();
    I const max_blocks = num_cu;

    bool const is_tall = (nrows >= ncols);
    I nbx = (is_tall) ? std::min(ceil(nrows, nx), num_cu) : 1;
    I nby = (is_tall) ? 1 : std::min(ceil(ncols, ny), num_cu);

    I const nbz = std::min(max_blocks, batch_count);
    if(batch_count >= num_cu)
    {
        // -----------------------------------------
        // large batch_count, assign 1 thread block
        // per batch entry matrix
        // -----------------------------------------
        nbx = 1;
        nby = 1;
    }

    amax_matrix<S, I, Istride, UA>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), lds_size, stream>>>(nrows, ncols, p_amax,

                                                                     A_, shift_A, ldA, stride_A,

                                                                     batch_count, work);
}

ROCSOLVER_END_NAMESPACE
