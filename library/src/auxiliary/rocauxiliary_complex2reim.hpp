
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

// ----------------------------------------------------------------------------
// kernel to split a complex matrix into the real part matrix and imaginary part matrix
// need some scratch space work   of size   nrows * nb,
// where nb is number of blocks in y-dimension
// if LDS is used, then work space is not required
//
//
// NOTE: assume one thread block handle full column
// ----------------------------------------------------------------------------
template <typename T, typename I, typename Istride, typename UA, typename UA_re, typename UA_im>
static __global__ complex2reim_inplace_kernel(

    I const nrows,
    I const ncols,

    UA A_,
    Istride const shift_A,
    I const ldA,
    Istride const stride_A,

    UA_re A_re_,
    Istride const shift_A_re,
    I const ldA_re,
    Istride const stride_A_re,

    UA_im A_im_,
    Istride const shift_A_im,
    I const ldA_im,
    Istride const stride_A_im,

    I const batch_count,

    T* const work,
    I const lds_size)
{
    assert(rocblas_is_complex<T>);
    using S = decltype(std::real(T{}));

    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    extern __shared__ double ldsmem[];

    bool const fit_in_lds = ((sizeof(T) * n) <= lds_size);

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const j_start = blockIdx.y;
    I const j_inc = gridDim.y;

    I const i_start = threadIdx.x;
    I const i_inc = blockDim.x;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + (j * static_cast<int64_t>(ld))); };

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        auto const A_p = load_ptr_batch(A_, bid, shift_A, stride_A);
        auto const A_re_p = load_ptr_batch(A_re_, bid, shift_A_re, stride_A_re);
        auto const A_im_p = load_ptr_batch(A_im_, bid, shift_A_im, stride_A_im);

        T* const p_work = (T*)work;
        T* const p_ldsmem = (T*)&(lds[0]);

        T* const Aj_ = (fit_in_lds) ? p_ldsmem : p_work + j_start * nrows;

        auto Aj = [=](auto i) -> T& { return (Aj_[i]); };

        auto A = [=](auto i, auto j) -> T& { return (A_p[idx2D(i, j, ldA)]); };

        for(auto j = j_start; j < ncols; j += j_inc)
        {
            // ------------------------------------------
            // read entire column into scratch array Aj(:)
            // ------------------------------------------
            for(auto i = i_start; i < nrows; i += i_inc)
            {
                Aj(i) = A(i, j);
            }
            __syncthreads();

            // ---------------------------------------------------
            // split and store Aj(:) into real part and imaginary part
            // note: it will still work if A_re(:,:) and A_im(:,:)
            // over-writes original A(:,:)
            // ---------------------------------------------------
            for(auto i = i_start; i < nrows; i += i_inc)
            {
                auto const aij = Aj(i);

                A_re_p[idx2D(i, j, ldA_re)] = aij.real();
                A_im_p[idx2D(i, j, ldA_im)] = aij.imag();
            }
            __syncthreads();

        } // end for j

    } // end for bid
}

// ----------------------------------------------------------------------------
// kernel to split a complex matrix into the real part matrix and imaginary part matrix
//
// NOTE: assume no overlap in A, A_re, A_im
// ----------------------------------------------------------------------------
template <typename T, typename I, typename Istride, typename UA, typename UA_re, typename UA_im>
static __global__ complex2reim_outofplace_kernel(

    I const nrows,
    I const ncols,

    UA A_,
    Istride const shift_A,
    I const ldA,
    Istride const stride_A,

    UA_re A_re_,
    Istride const shift_A_re,
    I const ldA_re,
    Istride const stride_A_re,

    UA_im A_im_,
    Istride const shift_A_im,
    I const ldA_im,
    Istride const stride_A_im,

    I const batch_count

)
{
    assert(rocblas_is_complex<T>);
    using S = decltype(std::real(T{}));

    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const i_inc = blockDim.x * gridDim.x;
    I const j_inc = blockDim.y * gridDim.y;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + (j * static_cast<int64_t>(ld))); };

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        auto const A_p = load_ptr_batch(A_, bid, shift_A, stride_A);
        auto const A_re_p = load_ptr_batch(A_re_, bid, shift_A_re, stride_A_re);
        auto const A_im_p = load_ptr_batch(A_im_, bid, shift_A_im, stride_A_im);

        for(auto j = j_start; j < ncols; j += j_inc)
        {
            for(auto i = i_start; i < nrows; i += i_inc)
            {
                auto const aij = A_p[idx2D(i, j, ldA)];

                A_re_p[idx2D(i, j, ldA_re)] = aij.real();
                A_im_p[idx2D(i, j, ldA_im)] = aij.imag();
            }

        } // end for j

    } // end for bid
}

// -------------------------------------------------------
// function to split a nrows by ncols complex matrix into
// the real matrix part and imaginary matrix part
// note: it is possible to over-write original matrix
// as (2 * nrows) by ncols  real matrix
// [ A_re ]
// [ A_im ]
// -------------------------------------------------------
template <typename T, typename I, typename Istride, typename UA, typename UA_re, typename UA_im>
static void complex2reim_inplace(hipStream_t stream,
                                 I const nrows,
                                 I const ncols,

                                 UA A_,
                                 Istride const shift_A,
                                 I const ldA,
                                 Istride const stride_A,

                                 UA_re A_re_,
                                 Istride const shift_A_re,
                                 I const ldA_re,
                                 Istride const stride_A_re,

                                 UA_im A_im_,
                                 Istride const shift_A_im,
                                 I const ldA_im,
                                 Istride const stride_A_im,

                                 I const batch_count,
                                 T* const work,
                                 size_t const lwork_in_bytes)
{
    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    I const lds_size = get_lds_size();

    I const num_cu = get_num_cu();
    I const max_blocks = num_cu;
    I const num_cols = lwork_in_bytes / (sizeof(T) * nrows);

    bool const fit_in_lds = ((sizeof(T) * nrows) <= lds_size);
    // -------------------------------------------------------
    // if we can use lds and not touch work space
    // we can launch more thread blocks for higher concurrency
    // -------------------------------------------------------
    I const nby = (fit_in_lds) ? std::min(max_blocks, ncols)
                               : std::max(I{1}, std::min(num_cu, std::min(num_cols, ncols)));

    I const nbz = (fit_in_lds) ? std::min(max_blocks, batch_count) : 1;

    I const nbx = 1;

    // ------------------------------------------
    // assume one thread block handle full column
    // ------------------------------------------
    I const num_threads = get_max_threads();
    I const nx = std::min(num_threads, nrows);

    complex2reim_inplace_kernel<T, I, Istride, UA, UA_re, UA_im>
        <<<dim3(nbx, nby, nbz), dim3(nx, 1, 1), lds_size, stream>>>(

            nrows, ncols,

            A_, shift_A, ldA, stride_A,

            A_re_, shift_A_re, ldA_re, stride_A_re,

            A_im_, shift_A_im, ldA_im, stride_A_im,

            batch_count, work, lds_size);
}

// -------------------------------------------------------
// function to split a nrows by ncols complex matrix into
// the real matrix part and imaginary matrix part
//
// Note: assume no overlap in A, A_re, A_im
// -------------------------------------------------------
template <typename T, typename I, typename Istride, typename UA, typename UA_re, typename UA_im>
static void complex2reim_outofplace(hipStream_t stream,
                                    I const nrows,
                                    I const ncols,

                                    UA A_,
                                    Istride const shift_A,
                                    I const ldA,
                                    Istride const stride_A,

                                    UA_re A_re_,
                                    Istride const shift_A_re,
                                    I const ldA_re,
                                    Istride const stride_A_re,

                                    UA_im A_im_,
                                    Istride const shift_A_im,
                                    I const ldA_im,
                                    Istride const stride_A_im,

                                    I const batch_count)
{
    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    I const num_cu = get_num_cu();
    I const max_blocks = num_cu;

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    I const nx = 32;
    I const ny = 32;
    I const nbx = std::min(max_blocks, ceil(nrows, nx));
    I const nby = std::min(max_blocks, ceil(ncols, ny));
    I const nbz = std::min(max_blocks, batch_count);

    complex2reim_outofplace_kernel<T, I, Istride, UA, UA_re, UA_im>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(

            nrows, ncols,

            A_, shift_A, ldA, stride_A,

            A_re_, shift_A_re, ldA_re, stride_A_re,

            A_im_, shift_A_im, ldA_im, stride_A_im,

            batch_count);
}

// -------------------------------------------
// Estimate the amount of scratch space needed
// -------------------------------------------
template <typename T, typename I>
static void complex2reim_inplace_getMemorySize(I const nrows,
                                               I const ncols,

                                               I const batch_count,
                                               size_t* p_lwork_in_bytes)
{
    auto const num_cu = get_num_cu();
    size_t const lwork_in_bytes = sizeof(T) * nrows * num_cu;

    auto const lds_size = get_lds_size();
    bool const fit_in_lds = ((sizeof(T) * nrows) <= lds_size);
    *p_lwork_in_bytes = (fit_in_lds) ? 0 : lwork_in_bytes;
}
ROCSOLVER_END_NAMESPACE
