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

#include "rocauxiliary_utility.hpp"

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

// ---------------------------------
// compute norm of a matrix
// norm == 'M' or 'm', max( abs(A(i,j)) )
// norm == '1' or 'O' or 'o', norm1(A), max column sum
// norm == 'I' or 'i', normI(A), max row sum
// norm == 'F' or 'f', 'E' or 'e', sqrt of sum of squares
//
// NOTE: assume dnorm[] has been set to zero
// this is important for correctness
// ---------------------------------

template <typename I, typename Istride, typename UA>
__global__ static void lange_kernel(char const norm,
                                    I const m,
                                    I const n,
                                    UA A,
                                    Istride const shiftA,
                                    I const lda,
                                    Istride strideA,

                                    double dnorm[],
                                    I const batch_count,
                                    void* work,
                                    size_t size_work)
{
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    // max abs(A(i,j))
    bool const is_norm_M = (norm == 'M') || (norm == 'm');

    // max column sum
    bool const is_norm_1 = (norm == '1') || (norm == 'O') || (norm == 'o');

    // max row sum
    bool const is_norm_I = (norm == 'I') || (norm == 'i');

    // sqrt of sum of squares
    bool const is_norm_F = (norm == 'F') || (norm == 'f') || (norm == 'E') || (norm == 'e');

    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_inc = hipBlockDim_x * hipGridDim_x;
    I const j_inc = hipBlockDim_y * hipGridDim_y;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const j_start = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    I const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    I const nthreads = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;

    extern __shared__ double sharedData[];

    auto sum_reduction = [=]() {
        // ---------------------------------
        // max reduction using shared memory
        // ---------------------------------
        for(int stride = nthreads / 2; stride > 0; stride /= 2)
        {
            if(tid < stride)
            {
                sharedData[tid] = (sharedData[tid] + sharedData[tid + stride]);
            }
            __syncthreads();
        }
        __syncthreads();
    };
    auto max_reduction = [=]() {
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
    };
    auto idx2D = [](auto i, auto j, auto ld) { return (i + (j * static_cast<int64_t>(ld))); };

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        auto __restrict__ Ap = load_ptr_batch(A_, bid, shiftA, strideA);
        auto A = [=](auto i, auto j) { return (Ap[idx2D(i, j, ldA)]); };

        if(is_norm_M)
        {
            // ----------------
            // max( abs(A(i,j))
            // ----------------
            double amax = 0;
            for(I j = j_start; j < n; j += j_inc)
            {
                for(I i = i_start; i < m; i += i_inc)
                {
                    amax = std::max(amax, std::abs(A(i, j)));
                }
            }

            sharedData[tid] = amax;
            __syncthreads();

            max_reduction();

            __syncthreads();
            if(tid == 0)
            {
                atomicMax(&(dnorm[bid]), sharedData[0]);
            }
            __syncthreads();
        }
        else if(is_norm_1)
        {
            // max column sum

            // ------------------------------------------------------------
            // launch as dim(1, nby,1), dim3(1024,1,1), sizeof(double)*1024
            // ------------------------------------------------------------
            double max_col_sum = 0;
            for(I j = j_start; j < n; j += j_inc)
            {
                double col_sum = 0;
                for(I i = tid; i < m; i += nthreads)
                {
                    col_sum += std::abs(A(i, j));
                }

                sharedData[tid] = col_sum;
                __syncthreads();

                sum_reduction();
                __syncthreads();

                if(tid == 0)
                {
                    max_col_sum = std::max(max_col_sum, sharedDatat[0]);
                }
            } // end for j
            __syncthreads();
            if(tid == 0)
            {
                atomicMax(&(dnorm[bid]), max_col_sum);
            }
            __syncthreads();
        }

    } // end for bid
}

template <typename I, typename Istride, typename AA, typename CC>
static void lacpy(hipStream_t stream,
                  char const uplo,
                  I const m,
                  I const n,
                  AA A,
                  Istride const shiftA,
                  I const lda,
                  Istride strideA,
                  CC C,
                  Istride const shiftC,
                  I const ldc,
                  Istride strideC,
                  I const batch_count)
{
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    I const max_blocks = 1024;
    I const nx = 32;
    I const ny = 32;
    I const nbx = std::min(max_blocks, ceil(m, nx));
    I const nby = std::min(max_blocks, ceil(n, ny));
    I const nyz = std::min(max_blocks, batch_count);

    lacpy_kernel<I, Istride, AA, CC>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(uplo, m, n,

                                                              A, shiftA, lda, strideA,

                                                              C, shiftC, ldc, strideC,

                                                              batch_count);
}

ROCSOLVER_END_NAMESPACE
