/*****************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"
#include <hip/hip_cooperative_groups.h>

ROCSOLVER_BEGIN_NAMESPACE

// ------------------------------------------------------
// kernel to implement    C <-  alpha * op(A) + beta * C
// where op(A) can be A, transpose(A) or conjugate_transpose(A)
//
// launch as dim3(nbx,nby,nbz), dim3(nx,ny,1)
// where nbx = ceil( nrowsC, nx )
//       nby = ceil( ncolsC, ny )
//       nbz = batch_count
// ------------------------------------------------------
template <typename T, typename I, typename Istride, typename UA, typename UC>
__global__ static void geadd_kernel(char trans,

                                    I const nrowsC,
                                    I const ncolsC,

                                    T const alpha,

                                    UA A_,
                                    Istride const shift_A,
                                    I const ld_A,
                                    Istride const stride_A,

                                    T const beta,

                                    UC C_,
                                    Istride const shift_C,
                                    I const ld_C,
                                    Istride const stride_C,

                                    I const batch_count)
{
    bool const has_work = (nrowsC >= 1) && (ncolsC >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;
    I const i_inc = blockDim.x * gridDim.x;
    I const j_inc = blockDim.y * gridDim.y;

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    bool const is_conj_transpose = (trans == 'C') || (trans == 'c');
    bool const is_transpose = (trans == 'N') || (trans == 'n');
    bool const is_no_transpose = (!is_transpose) && (!is_conj_transpose);

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    bool constexpr is_complex = rocblas_is_complex<T>;

    bool const zero = 0.0;
    bool const is_beta_zero = (beta == zero);

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        auto const Ap = load_ptr_batch(A_, bid, shift_A, stride_A);
        auto const Cp = load_ptr_batch(C_, bid, shift_C, stride_C);

        for(I j = j_start; j < ncolsC; j += j_inc)
        {
            for(I i = i_start; i < nrowsC; i += i_inc)
            {
                auto const ij_a = (is_no_transpose) ? idx2D(i, j, ld_A) : idx2D(j, i, ld_A);
                auto const ij_c = idx2D(i, j, ld_C);

                auto const aij = Ap[ij_a];
                T alpha_aij = zero;
                if constexpr(is_complex)
                {
                    alpha_aij = (is_conj_transpose) ? alpha * std::conj(aij) : alpha * aij;
                }
                else
                {
                    alpha_aij = alpha * aij;
                }

                if(is_beta_zero)
                {
                    Cp[ij_c] = zero;
                }
                else
                {
                    Cp[ij_c] *= beta;
                }
                Cp[ij_c] += alpha_aij;
            }
        }
    }
}

// ----------------------------------------------------------------
// routine to perform  C <- alpha * op(A) + beta * C
// where op(A) can be A, or transpose(A), or conjugate_transpose(A)
//
// this functionarlity is similar to PxGEADD in Parallel BLAS
// ----------------------------------------------------------------
template <typename T, typename I, typename Istride, typename UA, typename UC>
static rocblas_status rocsolver_geadd_template(rocblas_handle handle,
                                               char const trans,

                                               I const nrowsC,
                                               I const ncolsC,

                                               T const alpha,

                                               UA A_,
                                               Istride const shift_A,
                                               I const ld_A,
                                               Istride const stride_A,

                                               T const beta,

                                               UC C_,
                                               Istride const shift_C,
                                               I const ld_C,
                                               Istride const stride_C,

                                               I const batch_count)
{
    bool const has_work = (nrowsC >= 1) && (ncolsC >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);

    auto ceil = [](auto n, auto b) { return (((n - 1) / b) + 1); };

    I const max_blocks = 1024;
    I const nx = 32;
    I const ny = 32;

    I const nbx = std::min(max_blocks, ceil(nrowsC, nx));
    I const nby = std::min(max_blocks, ceil(ncolsC, ny));
    I const nbz = std::min(max_blocks, batch_count);

    geadd_kernel<T, I, Istride, UA, UC>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(trans,

                                                              nrowsC, ncolsC,

                                                              alpha,

                                                              A_, shift_A, ld_A, stride_A,

                                                              beta,

                                                              C_, shift_C, ld_C, stride_C,

                                                              batch_count);

    rocblas_set_pointer_mode(handle, old_mode);

    return (rocblas_status_success);
}

ROCSOLVER_END_NAMESPACE
