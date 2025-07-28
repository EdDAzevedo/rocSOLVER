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

// -------------------------------------------------------------
// scale and convert C(i,j,bid) = dlimit * (A(i,j,bid)/amax(bid))
//
// launch as dim3(nbx,nby,nbz), dim3(nx,ny,1)
// where
// nbx = min( max_blocks, ceil(m, nx))
// nby = min( max_blocks, ceil(n, ny))
// nbz = min( max_blocks, batch_count)
// -------------------------------------------------------------
template <typename Tfull, typename Treduced, typename Tscale, typename I, typename Istride>
static __global__ void scale_and_convert_kernel(I const nrows,
                                                I const ncols,

                                                Tfull* const A_,
                                                Istride const shift_A,
                                                I const ldA,
                                                Istride const stride_A,

                                                Treduced* const C_,
                                                Istride const shift_C,
                                                I const ldC,
                                                Istride const stride_C,

                                                I const batch_count,
                                                Tscale const dlimit,
                                                Tscale* p_amax

)
{
    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    I const i_inc = blockDim.x * gridDim.x;
    I const j_inc = blockDim.y * gridDim.y;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + (j * static_cast<int64_t>(ld))); };

    Tscale const zero = 0;
    Tscale const one = 1;

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        Tscale const amax = (p_amax == nullptr) ? one : p_amax[bid];
        Tscale const inv_amax = (amax == zero) ? one : one / amax;

        Tfull const* const __restrict__ A_p = load_ptr_batch(A_, bid, shift_A, stride_A);
        Treduced* const __restrict__ C_p = load_ptr_batch(C_, bid, shift_C, stride_C);

        for(I j = j_start; j < ncols; j += j_inc)
        {
            for(I i = i_start; i < nrows; i += i_inc)
            {
                auto const ij_a = idx2D(i, j, ldA);

                auto const aij = A_p[ij_a];
                auto const cij = (aij * inv_amax) * dlimit;

                auto const ij_c = idx2D(i, j, ldC);
                C_p[ij_c] = (Treduced)cij;
            }
        }
    }
}

//  ---------------------------------------------------------
//  scale and convert from A(0:(nrows-1), 0:(ncols-1))
//  to C(:, :) =    dlimit .* ( (1/amax) * A(:,:) )
//  ---------------------------------------------------------
template <typename Tfull, typename Treduced, typename Tscale, typename I, typename Istride>
static void scale_and_convert(rocblas_handle handle,
                              I const nrows,
                              I const ncols,

                              Tfull* const A_,
                              Istride const shift_A,
                              I const ldA,
                              Istride const stride_A,

                              Treduced* const C_,
                              Istride const shift_C,
                              I const ldC,
                              Istride const stride_C,

                              I const batch_count,
                              Tscale const dlimit,
                              Tscale* const p_amax

)
{
    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const nx = 32;
    I const ny = 32;

    I const max_blocks = 1024;

    I const nbx = std::min(max_blocks, ceil(nrows, nx));
    I const nby = std::min(max_blocks, ceil(ncols, ny));
    I const nbz = std::min(max_blocks, batch_count);

    scale_and_convert_kernel<Tfull, Treduced, Tscale, I, Istride>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(nrows, ncols,

                                                              A_, shift_A, ldA, stride_A,

                                                              C_, shift_C, ldC, stride_C,

                                                              batch_count, dlimit, p_amax

        );
}
ROCSOLVER_END_NAMESPACE
