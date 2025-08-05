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

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

ROCSOLVER_BEGIN_NAMESPACE

#ifndef HIP_CHECK
#define HIP_CHECK(fcn)               \
    {                                \
        auto const istat = (fcn);    \
        assert(istat == hipSuccess); \
    }
#endif

// ---------------------------------
// copy lower or upper or full
// m by n submatrix from A to C
//
// this routine can also be used for type conversion
// with clamp to avoid Inf values
//
//
// launch as
// dim3(nbx,nby,nbz), dim3(nx,ny,1)
// where
// nbx = min( max_blocks, ceil( m, nx ))
// nby = min( max_blocks, ceil( n, ny ))
// nbz = min( max_blocks, batch_count)
// ---------------------------------

template <typename I, typename Istride, typename AA, typename CC>
__global__ static void lacpy_kernel(char const uplo,
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

    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_inc = hipBlockDim_x * hipGridDim_x;
    I const j_inc = hipBlockDim_y * hipGridDim_y;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const j_start = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    bool const use_upper = (uplo == 'U') || (uplo == 'u');
    bool const use_lower = (uplo == 'L') || (uplo == 'l');
    bool const use_all = (!use_upper) && (!use_lower);

    auto idx2D = [](auto i, auto j, auto ld) { return (i + (j * static_cast<int64_t>(ld))); };

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        auto const Ap = load_ptr_batch(A, bid, shiftA, strideA);
        auto const Cp = load_ptr_batch(C, bid, shiftC, strideC);

        using Tf = decltype(*Ap);
        using Tr = decltype(*Cp);
        using Sr = decltype(std::real(*Cp));
        bool const is_complex = rocblas_is_complex<Tf>;
        assert(rocblas_is_complex<Tf> == rocblas_is_complex<Tr>);

        // -------------------------
        // clamp values to avoid Inf
        // -------------------------
        double dlimit = std::numeric_limits<Sr>::max();

        if(use_all)
        {
            for(I j = j_start; j < n; j += j_inc)
            {
                for(I i = i_start; i < m; i += i_inc)
                {
                    auto const ij_a = idx2D(i, j, lda);
                    auto const aij = Ap[ij_a];
                    auto const ij_c = idx2D(i, j, ldc);

                    if constexpr(is_complex)
                    {
                        auto const aij_real = std::real(aij);
                        auto const aij_imag = std::imag(aij);

                        Sr const cij_real = std::clamp(aij_real, -dlimit, dlimit);
                        Sr const cij_imag = std::clamp(aij_imag, -dlimit, dlimit);

                        Cp[ij_c] = Tr{cij_real, cij_imag};
                    }
                    else
                    {
                        Sr const cij = std::clamp(aij, -dlimit, dlimit);
                        Cp[ij_c] = cij;
                    }
                }
            }
        }
        else
        {
            for(I j = j_start; j < n; j += j_inc)
            {
                for(I i = i_start; i < m; i += i_inc)
                {
                    bool const do_assign = (use_upper && (i <= j)) || (use_lower && (i >= j));

                    if(do_assign)
                    {
                        auto const ij_c = idx2D(i, j, ldc);
                        auto const ij_a = idx2D(i, j, lda);
                        auto const aij = Ap[ij_a];

                        if constexpr(is_complex)
                        {
                            auto const aij_real = std::real(aij);
                            auto const aij_imag = std::imag(aij);

                            auto const cij_real = std::clamp(aij_real, -dlimit, dlimit);
                            auto const cij_imag = std::clamp(aij_imag, -dlimit, dlimit);

                            Cp[ij_c] = Tr{cij_real, cij_imag};
                        }
                        else
                        {
                            Sr const cij = std::clamp(aij, -dlimit, dlimit);
                            Cp[ij_c] = cij;
                        }
                    }
                }
            }
        }
    }
}

template <typename I, typename Istride, typename AA, typename CC>
static void lacpy(rocblas_handle handle,
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

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    I const max_blocks = 1024;
    I const nx = 32;
    I const ny = 32;
    I const nbx = std::min(max_blocks, ceil(m, nx));
    I const nby = std::min(max_blocks, ceil(n, ny));
    I const nbz = std::min(max_blocks, batch_count);

    lacpy_kernel<I, Istride, AA, CC>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(uplo, m, n,

                                                              A, shiftA, lda, strideA,

                                                              C, shiftC, ldc, strideC,

                                                              batch_count);
}

ROCSOLVER_END_NAMESPACE
