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

template <typename Tfull, typename Treduced, typename I, typename Istride>
static __global__ void scale_and_convert_kernel(I const nrows,
                                                I const ncols,
                                                Tfull const dlimit,
                                                Tfull* p_amax,
                                                Istride const strideP,

                                                Tfull* const A_,
                                                Istride const shift_A,
                                                I const ldA,
                                                Istride const stride_A,

                                                Treduced* const C_,
                                                Istride const shift_C,
                                                I const ldC,
                                                Istride const stride_C,

                                                I const batch_count)
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

    Tfull const zero = 0;
    Tfull const one = 1;

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        Tfull const amax = (p_amax == nullptr) ? one : p_amax[bid * strideP];
        bool const amax_is_one = (amax == one);
        bool const dlimit_is_one = (dlimit == one);
        auto const inv_amax = (amax == zero) ? one : one / amax;

        auto const A_p = load_ptr_batch(A_, bid, shift_A, stride_A);
        auto const C_p = load_ptr_batch(C_, bid, shift_C, stride_C);

        for(auto j = j_start; j < ncols; j += j_inc)
        {
            for(auto i = i_start; i < nrows; i += i_inc)
            {
                auto const ij_a = idx2D(i, j, ldA);
                auto const ij_c = idx2D(i, j, ldC);

                auto const aij = A_p[ij_a];
                auto const aij_normalized = (amax_is_one) ? aij : inv_amax * aij;
                auto const cij = (dlimit_is_one) ? aij_normalized : dlimit * aij_normalized;

                C_p[ij_c] = (Treduced)cij;
            }
        }
    }
}

//  ---------------------------------------------------------
//  scale and convert from A(0:(nrows-1), 0:(ncols-1))
//  to C(:, :) =    dlimit .* ( (1/amax) * A(:,:) )
//  ---------------------------------------------------------
template <typename Tfull, typename Treduced, typename I, typename Istride>
static void scale_and_convert(hipStream_t stream,
                              I const nrows,
                              I const ncols,
                              Tfull* p_amax,
                              Istride const strideP,

                              Tfull* const A_,
                              Istride const shift_A,
                              I const ldA,
                              Istride const stride_A,

                              Treduced* const C_,
                              Istride const shift_C,
                              I const ldC,
                              Istride const stride_C,

                              I const batch_count)
{
    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    I const nx = 32;
    I const ny = 32;

    I const num_cu = get_num_cu();
    I const max_blocks = num_cu;

    I nbx = std::min(max_blocks, ceil(nrows, nx));
    I nby = std::min(max_blocks, ceil(ncols, ny));
    I nbz = std::min(max_blocks, batch_count);

    scale_and_convert_kernel<Tfull, Treduced, I, Istride>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(nrows, ncols, dlimit,

                                                              p_amax, strideP,

                                                              A_, shift_A, ldA, stride_A,

                                                              C_, shift_C, ldC, stride_C,

                                                              batch_count);
}
ROCSOLVER_END_NAMESPACE
