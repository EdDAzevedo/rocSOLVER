
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

ROCSOLVER_BEGIN_NAMESPACE

#ifndef HIP_CHECK
#define HIP_CHECK(fcn)               \
    {                                \
        auto const istat = (fcn);    \
        assert(istat == hipSuccess); \
    }
#endif

// ----------------------------------------------------------------------------
// kernel to create a complex matrix by merging
// the real part matrix and imaginary part matrix
//
// the amount of scratch space work is  of
// nbytes = (sizeof(T) *  nrows * nby) * bid_inc,
// where nby is number of blocks in y-dimension
//
// if then entire column can fit in LDS,
// then work space is not required
//
// NOTE: assume one thread block handle a full column
// ----------------------------------------------------------------------------
template <typename T, typename I, typename Istride, typename UA_re, typename UA_im, typename UA>
static __global__ void reim2complex_inplace_kernel(

    I const nrows,
    I const ncols,

    UA_re A_re_,
    Istride const shift_A_re,
    I const ldA_re,
    Istride const stride_A_re,

    UA_im A_im_,
    Istride const shift_A_im,
    I const ldA_im,
    Istride const stride_A_im,

    UA A_,
    Istride const shift_A,
    I const ldA,
    Istride const stride_A,

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

    bool const fit_in_lds = ((sizeof(T) * nrows) <= lds_size);

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
        T* const p_ldsmem = (T*)&(ldsmem[0]);

        T* const Aj_ = (fit_in_lds) ? p_ldsmem : p_work + nrows * idx2D(j_start, bid_start, j_inc);

        auto Aj = [=](auto i) -> T& { return (Aj_[i]); };

        auto A = [=](auto i, auto j) -> T& { return (A_p[idx2D(i, j, ldA)]); };

        for(auto j = j_start; j < ncols; j += j_inc)
        {
            // ------------------------------------------
            // read entire column of A_re(:,j) and A_im(:,j)
            // into scratch array Aj(:)
            // ------------------------------------------
            for(auto i = i_start; i < nrows; i += i_inc)
            {
                auto const aij_re = A_re_p[idx2D(i, j, ldA_re)];
                auto const aij_im = A_im_p[idx2D(i, j, ldA_im)];

                Aj(i) = T{aij_re, aij_im};
            }
            __syncthreads();

            // ---------------------------------------------------
            // merge Aj(:) into real part and imaginary part
            // note: it will still work if A_re(:,:) and A_im(:,:)
            // over-writes original A(:,:)
            // ---------------------------------------------------
            for(auto i = i_start; i < nrows; i += i_inc)
            {
                auto const aij = Aj(i);
                A(i, j) = aij;
            }
            __syncthreads();

        } // end for j

    } // end for bid
}
// ----------------------------------------------------------------------------
// kernel to create a complex matrix by merging
// the real part matrix and imaginary part matrix
//
// note: no scratch space is needed
// assume there is no overlap in the A, A_re, A_im arrays
// ----------------------------------------------------------------------------
template <typename I, typename Istride, typename UA_reim, typename UA>
static __global__ void reim2complex_outofplace_simple_kernel(

    I const nrows,
    I const ncols,

    UA_reim A_re_,
    Istride const shift_A_re,
    I const ldA_re,
    Istride const stride_A_re,

    UA_reim A_im_,
    Istride const shift_A_im,
    I const ldA_im,
    Istride const stride_A_im,

    UA A_,
    Istride const shift_A,
    I const ldA,
    Istride const stride_A,

    I const batch_count

)
{
    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    extern __shared__ double ldsmem[];

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

        using T = decltype(*A_p);
        bool constexpr is_complex
            = rocblas_is_complex<T> || std::is_same<UA, rocblas_float_complex*>::value
            || std::is_same<UA, rocblas_double_complex*>::value
            || std::is_same<UA, rocblas_float_complex**>::value
            || std::is_same<UA, rocblas_double_complex**>::value;
        assert(is_complex);

        for(auto j = j_start; j < ncols; j += j_inc)
        {
            for(auto i = i_start; i < nrows; i += i_inc)
            {
                auto const ij_a = idx2D(i, j, ldA);

                if constexpr(is_complex)
                {
                    auto const aij_re = A_re_p[idx2D(i, j, ldA_re)];
                    auto const aij_im = A_im_p[idx2D(i, j, ldA_im)];

                    // -----------------------------------------------
                    // TODO: why T aij{aij_re, aij_im} does not work
                    // -----------------------------------------------
                    std::complex<float> aij{aij_re, aij_im};
                    A_p[ij_a] = aij;
                }
                else
                {
                    A_p[ij_a] = A_re_p[idx2D(i, j, ldA_re)];
                }
            } // end for i
        } // end for j
    } // end for bid
}

// -------------------------------------------------------
// function to create a nrows by ncols complex matrix by merging
// the real matrix part and imaginary matrix part
// note: it is possible to over-write original matrix
// as (2 * nrows) by ncols  real matrix
// [ A_re ]
// [ A_im ]
// -------------------------------------------------------
template <typename I, typename Istride, typename UA_reim, typename UA>
static void reim2complex_inplace(hipStream_t stream,
                                 I const nrows,
                                 I const ncols,

                                 UA_reim A_re_,
                                 Istride const shift_A_re,
                                 I const ldA_re,
                                 Istride const stride_A_re,

                                 UA_reim A_im_,
                                 Istride const shift_A_im,
                                 I const ldA_im,
                                 Istride const stride_A_im,

                                 UA A_,
                                 Istride const shift_A,
                                 I const ldA,
                                 Istride const stride_A,

                                 I const batch_count,
                                 void* const work_arg,
                                 size_t const lwork_in_bytes)
{
    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    I const bid = 0;
    auto const Ap = load_ptr_batch(A_, bid, shift_A, stride_A);
    using T = decltype(*Ap);

    T* const work = (T*)work_arg;

    I const lds_size = get_lds_size();

    I const num_cu = get_num_cu();
    I const max_blocks = num_cu;
    I const num_cols = lwork_in_bytes / (sizeof(T) * nrows);

    bool const fit_in_lds = ((sizeof(T) * nrows) <= lds_size);
    I const nby = (fit_in_lds) ? std::min(max_blocks, ncols)
                               : std::max(I{1}, std::min(num_cu, std::min(num_cols, ncols)));

    I const nbz = (fit_in_lds) ? std::min(max_blocks, batch_count) : 1;

    I const nbx = 1;

    // ------------------------------------------
    // assume one thread block handle full column
    // ------------------------------------------
    I const num_threads = get_max_threads();
    I const nx = std::min(num_threads, nrows);

    reim2complex_inplace_kernel<I, Istride, UA_reim, UA>
        <<<dim3(nbx, nby, nbz), dim3(nx, 1, 1), lds_size, stream>>>(

            nrows, ncols,

            A_re_, shift_A_re, ldA_re, stride_A_re,

            A_im_, shift_A_im, ldA_im, stride_A_im,

            A_, shift_A, ldA, stride_A,

            batch_count, work, lds_size);
}

// -------------------------------------------------------
// function to create a nrows by ncols complex matrix by merging
// the real matrix part and imaginary matrix part
//
// note: assume A, A_re, A_im don't overlap
// -------------------------------------------------------
template <typename I, typename Istride, typename UA_reim, typename UA>
static void reim2complex_outofplace_simple(rocblas_handle handle,
                                           I const nrows,
                                           I const ncols,

                                           UA_reim A_re_,
                                           Istride const shift_A_re,
                                           I const ldA_re,
                                           Istride const stride_A_re,

                                           UA_reim A_im_,
                                           Istride const shift_A_im,
                                           I const ldA_im,
                                           Istride const stride_A_im,

                                           UA A_,
                                           Istride const shift_A,
                                           I const ldA,
                                           Istride const stride_A,

                                           I const batch_count)
{
    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
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

    I const nbx = std::min(max_blocks, ceil(nrows, nx));
    I const nby = std::min(max_blocks, ceil(ncols, ny));
    I const nbz = std::min(max_blocks, batch_count);

    reim2complex_outofplace_simple_kernel<I, Istride>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(

            nrows, ncols,

            A_re_, shift_A_re, ldA_re, stride_A_re,

            A_im_, shift_A_im, ldA_im, stride_A_im,

            A_, shift_A, ldA, stride_A,

            batch_count);
}

template <typename I, typename Istride, typename UA_reim, typename UA>
static void reim2complex_inplace_getMemorySize(I const nrows,
                                               I const ncols,

                                               UA_reim A_re_,
                                               Istride const shift_A_re,
                                               I const ldA_re,
                                               Istride const stride_A_re,

                                               UA_reim A_im_,
                                               Istride const shift_A_im,
                                               I const ldA_im,
                                               Istride const stride_A_im,

                                               UA A_,
                                               Istride const shift_A,
                                               I const ldA,
                                               Istride const stride_A,

                                               I const batch_count,
                                               size_t* p_lwork_in_bytes)
{
    *p_lwork_in_bytes = 0;
    bool const has_work = (nrows >= 1) && (ncols >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    I const bid = 0;
    auto const Ap = load_ptr_batch(A_, bid, shift_A, stride_A);
    using T = decltype(*Ap);

    auto const num_cu = get_num_cu();
    size_t const lwork_in_bytes = sizeof(T) * nrows * num_cu;

    auto const lds_size = get_lds_size();
    bool const fit_in_lds = ((sizeof(T) * nrows) <= lds_size);

    *p_lwork_in_bytes = (fit_in_lds) ? 0 : lwork_in_bytes;
}

template <typename Tcomplex, typename Treal, typename Tscale, typename I, typename Istride>
__global__ void complex2reim_inplace_kernel(const I m,
                                            const I n,
                                            const Tcomplex* A,
                                            const Istride shiftA,
                                            const I lda,
                                            const Istride strideA,
                                            Treal* A_re,
                                            const Istride shiftA_re,
                                            const I ld_re,
                                            const Istride strideA_re,
                                            Treal* A_im,
                                            const Istride shiftA_im,
                                            const I ld_im,
                                            const Istride strideA_im,
                                            const I batch_count,
                                            const Tscale dlimit,
                                            const Tscale* amax_re,
                                            const Tscale* amax_im,
                                            void* work,
                                            size_t size_work)
{
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    I const i_inc = blockDim.x * gridDim.x;
    I const j_inc = blockDim.y * gridDim.y;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;

    I const bid_inc = gridDim.z;
    I const bid_start = blockIdx.z;

    auto const ldA = lda;

    Tcomplex* pfree = (Tcomplex*)work;
    bool const isok = ((sizeof(Tcomplex) * m * j_inc * bid_inc) <= size_work);
    assert(isok);

    bool constexpr is_complex = rocblas_is_complex<Tcomplex>;

    extern __shared__ double ldsmem[];
    I const lds_size = 64 * 1024;
    bool const fit_in_lds = ((sizeof(Tcomplex) * m) <= lds_size);

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        Tcomplex* const A_bid = load_ptr_batch(A, bid, shiftA, strideA);
        Treal* const A_re_bid = load_ptr_batch(A_re, bid, shiftA_re, strideA_re);
        Treal* const A_im_bid = (is_complex && (A_im != nullptr))
            ? load_ptr_batch(A_im, bid, shiftA_im, strideA_im)
            : nullptr;

        Tcomplex* const Atmp_work = pfree + m * idx2D(j_start, bid_start, j_inc);
        Tcomplex* const Atmp_lds = (Tcomplex*)&(ldsmem[0]);
        Tcomplex* const Atmp = (fit_in_lds) ? Atmp_lds : Atmp_work;

        __syncthreads();

        for(I j = j_start; j < n; j += j_inc)
        {
            // -----------------------
            // save column j in Atmp[]
            // -----------------------
            for(I i = i_start; i < m; i += i_inc)
            {
                auto const ij_A = idx2D(i, j, ldA);

                auto const aij = A_bid[ij_A];

                Atmp[i] = aij;
            }
            __syncthreads();

            for(I i = i_start; i < m; i += i_inc)
            {
                auto const aij = Atmp[i];

                auto const ij_A_re = idx2D(i, j, ld_re);

                A_re_bid[ij_A_re] = std::real(aij);
                if constexpr(is_complex)
                {
                    auto const ij_A_im = idx2D(i, j, ld_im);
                    A_im_bid[ij_A_im] = std::imag(aij);
                }
            }
            __syncthreads();
        }
    } // end for bid
}

// --------------------------------------------------
// perform conversion but use the clamp to avoid
// generating overflow Inf values
//
// launch with dim3(nbx,nby,nbx), dim3(nx,ny,1)
// nbx = ceil(m, nx)
// nby = ceil(n, ny)
// nbz = batch_count
// --------------------------------------------------
template <typename Tcomplex, typename Treal, typename Tscale, typename I, typename Istride>
__global__ void complex2reim_clamp_kernel(I const m,
                                          I const n,

                                          Tcomplex const* const A,
                                          Istride const shiftA,
                                          I const ldA,
                                          Istride const strideA,

                                          Treal* const A_re,
                                          Istride const shiftA_re,
                                          I const ldA_re,
                                          Istride const strideA_re,

                                          Treal* const A_im,
                                          Istride const shiftA_im,
                                          I const ldA_im,
                                          Istride const strideA_im,

                                          const I batch_count,
                                          const Tscale dlimit)
{
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    // implement our own clamp that can run on device
    auto clamp = [](auto aij, auto amin, auto amax) {
        return ((aij < amin) ? amin : (aij > amax) ? amax : aij);
    };

    I const i_inc = blockDim.x * gridDim.x;
    I const j_inc = blockDim.y * gridDim.y;

    I const i_start = threadIdx.x + blockIdx.x * blockDim.x;
    I const j_start = threadIdx.y + blockIdx.y * blockDim.y;

    I const bid_inc = gridDim.z;
    I const bid_start = blockIdx.z;

    bool constexpr is_complex = rocblas_is_complex<Tcomplex>;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        Tcomplex const* const A_bid = load_ptr_batch(A, bid, shiftA, strideA);
        Treal* const A_re_bid = load_ptr_batch(A_re, bid, shiftA_re, strideA_re);
        Treal* const A_im_bid = (is_complex && (A_im != nullptr))
            ? load_ptr_batch(A_im, bid, shiftA_im, strideA_im)
            : nullptr;

        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = i_start; i < m; i += i_inc)
            {
                const auto ij_A = idx2D(i, j, ldA);
                const auto aij = A_bid[ij_A];
                const auto aij_re = std::real(aij);

                auto const ij_A_re = idx2D(i, j, ldA_re);
                A_re_bid[ij_A_re] = static_cast<Treal>(clamp(aij_re, -dlimit, dlimit));

                if constexpr(is_complex)
                {
                    if(A_im_bid != nullptr)
                    {
                        const auto ij_A_im = idx2D(i, j, ldA_im);
                        const auto aij_im = std::imag(aij);

                        A_im_bid[ij_A_im] = static_cast<Treal>(clamp(aij_im, -dlimit, dlimit));
                    }
                }
            }
        }

    } // end for bid
}

template <typename Tcomplex, typename Treal, typename Tscale, typename I, typename Istride>
void complex2reim_inplace(rocblas_handle handle,
                          const I m,
                          const I n,
                          const Tcomplex* A,
                          const Istride shiftA,
                          const I ldA,
                          const Istride strideA,
                          Treal* A_re,
                          const Istride shiftA_re,
                          const I ldA_re,
                          const Istride strideA_re,
                          Treal* A_im,
                          const Istride shiftA_im,
                          const I ldA_im,
                          const Istride strideA_im,
                          const I batch_count,
                          const Tscale dlimit,
                          const Tscale* amax_re,
                          const Tscale* amax_im,
                          void* work,
                          size_t size_work)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I const max_blocks = 1024;

    I const nx = std::min(m, get_max_threads());
    I const ny = 1;

    I const lds_size = get_lds_size();
    bool const fit_in_lds = (sizeof(Tcomplex) * m <= lds_size);

    I const nbx = 1;
    I nby = n;
    I nbz = 1;

    if(fit_in_lds)
    {
        nby = n;
        nbz = batch_count;
    }
    else
    {
        I num_cols = size_work / (sizeof(Tcomplex) * m);
        nby = std::min(num_cols, n);

        if(num_cols > n)
        {
            nbz = std::min(batch_count, num_cols / n);
        }
    }

    nby = std::min(max_blocks, nby);
    nbz = std::min(max_blocks, nbz);

    assert(sizeof(Tcomplex) * m * nby * nbz <= size_work);

    complex2reim_inplace_kernel<Tcomplex, Treal, Tscale, I, Istride>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), lds_size, stream>>>(
            m, n,

            A, shiftA, ldA, strideA,

            A_re, shiftA_re, ldA_re, strideA_re,

            A_im, shiftA_im, ldA_im, strideA_im,

            batch_count,

            dlimit, amax_re, amax_im,

            work, size_work);
}

template <typename Tcomplex, typename Treal, typename Tscale, typename I, typename Istride>
void complex2reim_clamp(rocblas_handle handle,
                        const I m,
                        const I n,

                        const Tcomplex* A,
                        const Istride shiftA,
                        const I lda,
                        const Istride strideA,

                        Treal* A_re,
                        const Istride shiftA_re,
                        const I ld_re,
                        const Istride strideA_re,

                        Treal* A_im,
                        const Istride shiftA_im,
                        const I ld_im,
                        const Istride strideA_im,

                        const I batch_count,
                        const Tscale dlimit)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    I const max_blocks = 1024;

    I const nx = 32;
    I const ny = 32;

    I const nbx = std::min(max_blocks, ceil(m, nx));
    I const nby = std::min(max_blocks, ceil(n, ny));
    I const nbz = std::min(max_blocks, batch_count);

    complex2reim_clamp_kernel<Tcomplex, Treal, Tscale, I, Istride>
        <<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(m, n,

                                                              A, shiftA, lda, strideA,

                                                              A_re, shiftA_re, ld_re, strideA_re,

                                                              A_im, shiftA_im, ld_im, strideA_im,

                                                              batch_count,

                                                              dlimit);
}
ROCSOLVER_END_NAMESPACE
