
/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "auxiliary/rocauxiliary_lacgv.hpp"
#include "auxiliary/rocauxiliary_larf.hpp"
#include "auxiliary/rocauxiliary_larfg.hpp"
#include "rocblas.hpp"
#include "roclapack_geqr2.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <typename T, typename I>
static rocblas_status rocblasCall_trmm_mem(rocblas_side const side,
                                           I const mm,
                                           I const nn,
                                           I const batch_count,
                                           size_t* size_trmm_byte)
{
    // -----------------------------------------------------
    // TODO: ** need to double check whether this is correct
    // -----------------------------------------------------
    *size_trmm_byte = sizeof(T*) * std::max(1, batch_count);
    return (rocblas_status_success);
}

// -----------------------------------------------
// copy vector
//
// launch as dim(nbocks,1,batch_count), dim3(nx,1,1)
// where nblocks = ceil( n, nx)
// -----------------------------------------------
template <typename T, typename I, typename Istride>
static __global__ void copy1D(rocblas_handle handle,
                              I const n,
                              T const* const x,
                              Istride const shiftx,
                              I const incx,
                              Istride stridex,

                              T* const y,
                              Istride const shifty,
                              I const incy,
                              Istride const stridey,
                              I const batch_count)
{
    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const xp = load_ptr_batch(x, bid, shiftx, stridex);
        T* const yp = load_ptr_batch(y, bid, shifty, stridey);

        for(I i = i_start; i < n; i += i_inc)
        {
            auto const ix = i * incx;
            auto const iy = i * incy;
            yp[iy] = xp[ix];
        }
    }
}

#ifndef RGEQR3_BLOCKSIZE
#define RGEQR3_BLOCKSIZE(T) \
    ((sizeof(T) == 4) ? 256 : (sizeof(T) == 8) ? 128 : (sizeof(T) == 16) ? 64 : 32)
#endif

template <bool BATCHED, typename T>
static void rocsolver_rgeqr3_getMemorySize(const rocblas_int m,
                                           const rocblas_int n,
                                           const rocblas_int batch_count,
                                           size_t* plwork_byte)
{
    assert(plwork_byte == nullptr);

    *plwork_byte = 0;
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        return;
    }

    const size_t nb = RGEQR3_BLOCKSIZE(T);
    size_t const size_T_byte = sizeof(T) * nb * nb * std::max(1, batch_count);
    size_t const size_W_byte = sizeof(T) * nb * nb * std::max(1, batch_count);
    size_t size_trmm_left_byte = 0;
    size_t size_trmm_right_byte = 0;
    {
    auto const istat_left = rocblasCall_trmm_mem<T>( rocblas_side_left, m,n,batch_count, &size_trmm_left_byte ;
    auto const istat_right = rocblasCall_trmm_mem<T>( rocblas_side_right, m,n,batch_count, &size_trmm_right_byte ;
    assert( istat_left == rocblas_status_success );
    assert( istat_right == rocblas_status_success );
    }

    size_t size_work_byte = 0;
    size_t size_norms_byte = 0;
    rocsolver_larfg_getMemorySize<T>(n, batch_count, &size_work_byte, &size_norms_byte);

    size_t const lwork_byte = (size_T_byte + size_W_byte)
        + (size_trmm_left_byte + size_trmm_right_byte) + (size_work_byte + size_norm2_byte);

    *plwork_bye = lwork_bye;
}

__device__ __host__ static double dconj(double x)
{
    return (x);
};

__device__ __host__ static float dconj(float x)
{
    return (x);
};

__device__ __host__ static std::complex<float> dconj(std::complex<float> x)
{
    return (std::conj(x));
};

__device__ __host__ static std::complex<double> dconj(std::complex<double> x)
{
    return (std::conj(x));
};

__device__ __host__ static rocblas_complex_num<float> dconj(rocblas_complex_num<float> x)
{
    return (conj(x));
};

__device__ __host__ static rocblas_complex_num<double> dconj(rocblas_complex_num<double> x)
{
    return (conj(x));
};

// -----------------------------------------
// geadd() performs matrix addition that is
// similar PxGEADD() in Parallel BLAS library
//
//  C(1:m,1:n) =  beta * C(1:m,1:n) + alpha * op(A)
//
// assume launch with
// dim3(nbx,nby,max_nblocks), dim3(nx,ny,1)
// -----------------------------------------
template <typename T, typename UA, typename UC, typename I, typename Istride>
static __global__ void geadd_kernel(char trans,
                                    I const m,
                                    I const n,
                                    I const batch_count,
                                    T const alpha,
                                    UA AA,
                                    I const shiftA,
                                    I const ldA,
                                    Istride const strideA,
                                    T const beta,
                                    UC CC,
                                    I const shiftC,
                                    I const ldC,
                                    Istride const strideC)
{
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    };

    auto const nbx = hipGridDim_x;
    auto const nby = hipGridDim_y;
    auto const nx = hipBlockDim_x;
    auto const ny = hipBlockDim_y;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * nx;
    I const i_inc = (nbx * nx);
    I const j_start = hipThreadIdx_y + hipBlockIdx_y * ny;
    I const j_inc = (nby * ny);

    bool const is_transpose = (trans == 'T' || trans == 't');
    bool const is_conj_transpose = (trans == 'C' || trans == 'c');

    auto const bid_inc = hipGridDim_z;
    auto const bid_start = hipBlockIdx_z;

    T const zero = 0;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const A = load_ptr_batch(AA, bid, shiftA, strideA);
        T* const C = load_ptr_batch(CC, bid, shiftC, strideC);

        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = i_start; i < m; i += i_inc)
            {
                auto const ij = idx2D(i, j, ldC);
                C[ij] = (beta == zero) ? zero : beta * C[ij];

                T const aij = (is_transpose) ? A[idx2D(j, i, ldA)]
                    : (is_conj_transpose)    ? dconj(A[idx2D(j, i, ldA)])
                                             : A[idx2D(i, j, ldA)];

                C[ij] += alpha * aij;
            }
        }
    }
}

template <typename T, typename UA, typename UC, typename I, typename Istride>
static void geadd_template(char trans,
                           I const m,
                           I const n,
                           I const batch_count,
                           T const alpha,
                           UA AA,
                           Istride const shiftA,
                           I const ldA,
                           Istride const strideA,
                           T const beta,
                           UC CC,
                           Istride const shiftC,
                           I const ldC,
                           Istride const strideC)
{
    auto ceil = [](auto m, auto nb) { return ((m - 1) / nb + 1); };

    auto min = [](auto x, auto y) { return ((x < y) ? x : y); };

    auto const max_threads = 1024;
    auto const max_blocks = 64 * 1000;

    auto const nx = 32;
    auto const ny = max_threads / nx;
    auto const nbx = min(max_blocks, ceil(m, nx));
    auto const nby = min(max_blocks, ceil(n, ny));
    auto const nbz = min(max_blocks, batch_count);

    geadd_kernel<T, UA, UC, I, Istride><<<dim3(nbx, nby, nbz), dim3(nx, ny, 1)>>>(
        trans, m, n, batch_count, alpha, AA, shiftA, ldA, strideA, beta, CC, shiftC, ldC, strideC);
}

// -------------------------------
//  [W] = formT3( Y1, T1, Y2, T2 )
//
//  compute T3 = W = -T1 * (Y1' * Y2 ) * T2
//
//  Y1 is m by k1
//  Y2 is (m-k1) by k2
//
//
//  Merged triangular matrix for block
//  Householder reflectors
//  H1 = (I - Y1 * T1 * Y1')
//  H2 = (I - Y2 * T2 * Y2')
//
//  H3 = H1 * H2 = (I - [Y1 | Y2] * Tmat * [Y1 | Y2]')
//  where
//  Tmat = [T1    T3]
//         [0     T2]
//  --------------------------------
template <bool BATCHED, typename T, typename I, typename U, typename Istride>
static rocblas_status formT3(rocblas_handle handle,
                             I const m,
                             I const k1,
                             I const k2,
                             I const batch_count,

                             U Y,
                             I const shiftY,
                             I const ldY,
                             Istride strideY,

                             T* const Tmat,
                             I const shiftT,
                             I const ldT,
                             Istride const strideT,
                             std::byte* dwork,
                             I const lwork_byte)
{
    std::byte* pfree = (std::byte*)dwork;

    Istride const T1_offset = idx2D(0, 0, ldT);
    Istride const T2_offset = idx2D(k1, k1, ldT);

    // --------------------------
    // ** Note ** reuse storage
    // k1 by  k2 matrix Wmat is
    //
    // Tmat( 0:(k1-1),  k1:(k1+k2-1) )
    // --------------------------
    auto Wmat = Tmat;
    Istride const shiftW = shiftT + idx2D(0, k1, ldT);
    auto const ldW = ldT;
    auto const strideW = strideT;

    auto const k = k1 + k2;

    Istride const Y11_offset = idx2D(0, 0, ldY);
    Istride const Y21_offset = idx2D(k1, 0, ldY);
    Istride const Y31_offset = idx2D(k, 0, ldY);

    Istride const Y12_offset = idx2D(k1, k1, ldY);
    Istride const Y22_offset = idx2D(k, k1, ldY);

    // -------------------------------
    //  [W] = formT3( Y1, T1, Y2, T2 )
    //
    //  compute T3 = W = -T1 * (Y1' * Y2 ) * T2
    //
    //  Y1 is m by k1
    //  Y2 is (m-k1) by k2
    //
    //  Let
    //
    //  Y1 = [ Y11 ]
    //       [ Y21 ]
    //       [ Y31 ]
    //
    //  Y11 is k1 by k1 unit lower diagonal but not used here
    //  Y21 is k2 by k1
    //  Y31 = (m-k) by k1,   where k = k1 + k2
    //
    //  Y2 = [   0 ]
    //       [ Y12 ]
    //       [ Y22 ]
    //
    //  Y12 is k2 by k2 unit lower diagonal
    //  Y22 is (m-k) by k2
    // -------------------------------

    auto const nrow_Y11 = k1;
    auto const ncol_Y11 = k1;

    auto const nrow_Y21 = k2;
    auto const ncol_Y21 = k1;

    auto const nrow_Y31 = (m - k);
    auto const ncol_Y31 = k1;

    auto const nrow_Y12 = k2;
    auto const ncol_Y12 = k2;

    auto const nrow_Y22 = (m - k);
    auto const ncol_Y22 = k2;

    auto const ldY1 = ldY;
    auto const ldY2 = ldY;

    auto const Y1_offset2 = idx2D(k1, k1, ldY2);
    auto const Y2_offset2 = idx2D(k1 + k2, k1, ldY2);

    // ------------------------
    // (0) first set W = Y21'
    // (1) TRMM   W = Y21' * Y12
    // (2) GEMM   W = W + Y31' * Y22
    // (3) TRMM   W = -T1 * W
    // (4) TRMM   W = W * T2
    // ------------------------

    {
        // ---------------------
        // (0) first set W = Y21'
        // ---------------------
        char const trans = 'C';
        I const mm = ncol_Y21;
        I const nn = nrow_Y21;
        T const alpha = 1;
        T const beta = 0;

        geadd_template(trans, mm, nn, batch_count, alpha, Y, shiftY + Y21_offset, ldY, strideY,
                       beta, Wmat, shiftW, ldW, strideW);
    }
    auto const nrow_W = ncol_Y21;
    auto const ncol_W = nrow_Y21;

    {
        // --------------------------------------
        // (1) TRMM   W = W * Y12, where W = Y21'
        //
        // NOTE:  Y12 is k2 by k2 lower unit diagonal
        // --------------------------------------
        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_lower;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_unit;
        I const mm = nrow_W;
        I const nn = ncol_W;

        T alpha = 1;
        Istride const stride_alpha = 0;

        size_t size_trmm_byte = 0;
        ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_byte));

        T* const work_Arr = (T*)pfree;
        pfree += size_trmm_byte;

        ROCBLAS_CHECK(rocblasCall_trmm<T>(handle, side, uplo, transA, diag, mm, nn, &alpha,
                                          stride_alpha, Y, shiftY + Y21_offset, ldY2, strideY, Wmat,
                                          shiftW, ldW, strideW, batch_count, work_Arr));
        pfree = pfree - size_trmm_byte;
    }

    {
        // -----------------------------
        // (2) GEMM   W = W + Y31' * Y22
        // -----------------------------

        rocblas_operation const trans_A = rocblas_operation_conjugate_transpose;
        rocblas_operation const trans_B = rocblas_operation_none;

        I const mm = nrow_W;
        I const nn = ncol_W;
        I const kk = nrow_Y22;

        T alpha = 1;
        T beta = 1;

        T** work = nullptr;
        ROCBLAS_CHECK(rocblasCall_gemm<T>(handle, trans_A, trans_B, mm, nn, kk, &alpha, Y,
                                          shiftY + Y31_offset, ldY, strideY, Y, shiftY + Y22_offset,
                                          ldY, strideY, &beta, Wmat, shiftW, ldW, strideW,
                                          batch_count, work));
    }

    {
        // ---------------------
        // (3) TRMM   W = -T1 * W
        // ---------------------
        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;
        I const mm = nrow_W;
        I const nn = ncol_W;
        T alpha = -1;
        Istride stride_alpha = 0;

        size_t size_trmm_byte = 0;
        ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_byte));

        T* work_Arr = (T*)pfree;
        pfree += size_trmm_byte;

        ROCBLAS_CHECK(rocblasCall_trmm<T>(handle, side, uplo, transA, diag, mm, nn, &alpha,
                                          stride_alpha, Tmat, shiftT + T1_offset, ldT, strideT,
                                          Wmat, shiftW, ldW, strideW, batch_count, work_Arr));

        pfree = pfree - size_trmm_byte;
    }

    {
        // ---------------------
        // (4) TRMM   W = W * T2
        // ---------------------

        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;
        I const mm = nrow_W;
        I const nn = ncol_W;
        T alpha = 1;
        Istride const stride_alpha = 0;

        size_t size_trmm_byte = 0;
        ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_byte));

        T* work_Arr = pfree;
        pfree += size_trmm_byte;

        ROCBLAS_CHECK(rocblasCall_trmm<T>(handle, side, uplo, transA, diag, mm, nn, &alpha,
                                          stride_alpha, Tmat, shiftT + T2_offset, ldT, strideT,
                                          Wmat, shiftW, ldW, strideW, batch_count, work_Arr));
        pfree = pfree - size_trmm_byte;
    }
    return (rocblas_status_success);
}

// --------------------------------------------------
// Algorithm inspired by the paper
// "Applying recursion to serial and parallel QR factorization"
// by Erik Elmroth and Fred G. Gustavson
// IBM Journal of Research and Development, August 2000
//
//
//  Input A(0:(m-1), 0:(n-1))
//  Output (Y, R, T),  Y and R replaces A, m >= n
//
//  R is n by n upper triangular
//  Y is lower trapezoidal with ones's on the diagonal
//  or Y(i,i) == 1, for i=0:(n-1)
//
//  Note T is n by n  upper triangular matrix
//  The diagonal entries T(i,i) are the "tau" values
//  in lapack GEQRF
// --------------------------------------------------

template <bool BATCHED, bool STRIDED, typename T, typename I, typename U, typename Istride, bool COMPLEX = rocblas_is_complex<T>>
static rocblas_status rocsolver_rgeqr3_template(rocblas_handle handle,
                                                const I m,
                                                const I n,
                                                const I batch_count,

                                                U A,
                                                const Istride shiftA,
                                                const I lda,
                                                const Istride strideA,

                                                T* const Tmat,
                                                const Istride shiftT,
                                                const I ldt,
                                                const Istride strideT,

                                                T* const Wmat,
                                                const Istride shiftW,
                                                const I ldw,
                                                const Istride strideW,

                                                std::byte* const dwork,
                                                I const lwork_byte_arg

)
{
    ROCSOLVER_ENTER("rgeqr3", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto lwork_byte = lwork_byte_arg;
    std::byte* pfree = (std::byte*)dwork;

    auto min = [](auto x, auto y) { return ((x < y) ? x : y); };
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    if(n == 1)
    {
        // ------------------------------------------------------
        // generate Householder reflector to work on first column
        // ------------------------------------------------------
        const I j = 0;
        Istride const strideT = (ldt * n);

        size_t size_work_byte = 0;
        size_t size_norms_byte = 0;
        rocsolver_larfg_getMemorySize<T>(n, batch_count, &size_work_byte, &size_norms_byte);

        {
            // ----------------
            // allocate scratch storage
            // ----------------

            T* const work = (T*)pfree;
            pfree += size_work_byte;
            lwork_byte = lwork_byte - size_work_byte;

            T* const norms = (T*)pfree;
            pfree += size_norms_byte;
            lwork_byte = lwork_byte - size_norms_byte;

            assert(lwork_byte >= 0);

            auto alpha = A;
            auto shifta = shiftA;
            auto x = A;
            auto shiftx = shiftA + 1;
            rocblas_int incx = 1;
            auto stridex = strideA;
            T* tau = Tmat + shiftT;
            auto strideP = strideT;

            rocsolver_larfg_template(handle, m, alpha, shifta, x, shiftx, incx, stridex, tau,
                                     strideP, batch_count, work, norms);

            // --------------------------
            // deallocate scratch storage
            // --------------------------
            pfree = pfree - size_work_byte;
            lwork_byte += size_work_byte;
            pfree = pfree - size_norms_byte;
            lwork_byte += size_norms_byte;
        }
    }
    else
    {
        // -----------------
        // perform recursion
        // -----------------
        auto const n1 = n / 2;
        auto const n2 = n - n1;
        auto const j1 = n1 + 1;

        auto const k1 = n1;
        auto const k2 = n2;

        // --------------------------------------------
        // [Y1, R1, T1 ] = rgeqr3( A(0:(m-1), 0:(n1-1))
        // --------------------------------------------

        {
            T* const T1 = Tmat;
            auto const mm = m;
            auto const nn = n1;
            ROCBLAS_CHECK(rocsolver_rgeqr3_template<BATCHED, STRIDED, T>(
                handle, mm, nn, batch_count, A, shiftA, lda, strideA, Tmat, shiftT, ldt, strideT,
                Wmat, shiftW, ldw, strideW, dwork, lwork_byte));
        }

        // -----------------------------------------------------
        // compute  A(0:(m-1), n1:(n-1) ) = Q1' * A( 0:(m-1), n1:n )
        //
        // where Q1 = eye - Y1 * T1 * Y1',
        // and Y1 is lower trapezoidal with unit diagonal
        // -----------------------------------------------------
        {
            // --------------------
            // get memory for larfb
            // --------------------
            T* tmptr = nullptr;
            T** workArr = nullptr;

            {
                size_t size_tmptr_byte = 0;
                size_t size_workArr_byte = 0;
                I const mm = m;
                I const nn = n - n1;
                I const kk = n1;
                rocsolver_larfb_getMemorySize<BATCHED, T>(rocblas_side_left, mm, nn, kk, batch_count,
                                                          &size_tmptr_byte, &size_workArr_byte);

                assert((size_tmptr_byte + size_workArr_byte) <= lwork_byte);

                T* const tmptr = (T*)pfree;
                pfree += size_tmptr_byte;
                lwork_byte = lwork_byte - size_tmptr_byte;
                T* const workArr = (T*)pfree;
                pfree += size_workArr_byte;
                lwork_byte = lwork_byte - size_workArr_byte;

                assert(lwork_byte >= 0);

                // -----------------------------------------------
                // call larfb to apply block Householder reflector
                // -----------------------------------------------

                const rocblas_side side = rocblas_side_left;
                const rocblas_operation trans = rocblas_operation_conjugate_transpose;
                const rocblas_direct direct = rocblas_forward_direction;
                const rocblas_storev storev = rocblas_column_wise;

                auto const Y = A;
                auto const shiftY = shiftA;
                auto const ldy = lda;
                auto const strideY = strideA;

                ROCBLAS_CHECK(rocsolver_larfb_template<BATCHED, STRIDED, T>(
                    handle, side, trans, direct, storev, mm, nn, kk, Y, shiftY, ldy, strideY, Tmat,
                    shiftT, ldt, strideT, A, shiftA + idx2D(0, n1, lda), lda, strideA, batch_count,
                    tmptr, workArr));
                // ------------------------
                // release memory for larfb
                // ------------------------

                pfree = pfree - size_tmptr_byte;
                lwork_byte += size_tmptr_byte;
                pfree = pfree - size_workArr_byte;
                lwork_byte += size_workArr_byte;
            }

            // -----------------------------------------
            // [Y2, R2, T2 ] = rgeqr3( A( n1:m, n1:n ) )
            // -----------------------------------------

            {
                auto const mm = (m - j1 + 1);
                auto const nn = (n - j1 + 1);
                auto const A2_offset = idx2D((j1 - 1), (j1 - 1), lda);
                auto const T2_offset = idx2D(k1, k1, ldt);

                ROCBLAS_CHECK(rocsolver_rgeqr3_template<BATCHED, STRIDED, T>(
                    handle, mm, nn, batch_count, A, shiftA + A2_offset, lda, strideA, Tmat,
                    shiftT + T2_offset, ldt, strideT, Wmat, shiftW, ldw, strideW, pfree, lwork_byte));
            }

            {
                // -------------------------------------------------------
                // compute T3 = T(0:(n1-1), n1:n ) = -T1 * (Y1' * Y2) * T2
                //
                // Note that
                // Y1 is m by n1 unit lower trapezoidal,
                // Y2 is (m-n1) by n2 lower trapezoidal
                // ------------------------------------
                auto Y = A;
                auto const shiftY = shiftA;
                auto const ldy = lda;
                auto const strideY = strideA;

                ROCBLAS_CHECK(formT3<BATCHED, T, I, U, Istride>(handle, m, n1, n2, batch_count, Y,
                                                                shiftY, ldy, strideY, Tmat, shiftT,
                                                                ldt, strideT, pfree, lwork_byte));
            }

            // --------------------------------------------------------------
            // implicitly form Y = [Y1, Y2] where Y is unit lower trapezoidal
            // Note Y over-writes lower part of A
            // --------------------------------------------------------------
            //

            // -----------------------------------
            // R = [ R1     A(0:(n1-1), n1:(n-1)) ]
            //     [ 0      R2                    ]
            //
            // Note R is n by n upper triangular
            // and over-writes matrix A
            // -----------------------------------

            // -----------------------------------
            // T = [ T1     T3 ]
            //     [ 0      T2 ]
            // -----------------------------------
        }
    }
    return (rocblas_status_success);
}
#if(0)

for(rocblas_int j = 0; j < dim; ++j)
{
    // generate Householder reflector to work on column j
    rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(j, j, lda), A,
                             shiftA + idx2D(std::min(j + 1, m - 1), j, lda), 1, strideA, (ipiv + j),
                             strideP, batch_count, (T*)work_workArr, Abyx_norms);

    // insert one in A(j,j) tobuild/apply the householder matrix
    ROCSOLVER_LAUNCH_KERNEL(set_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream, diag, 0,
                            1, A, shiftA + idx2D(j, j, lda), lda, strideA, 1, true);

    // conjugate tau
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, 1, ipiv, j, 1, strideP, batch_count);

    // Apply Householder reflector to the rest of matrix from the left
    if(j < n - 1)
    {
        rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                shiftA + idx2D(j, j, lda), 1, strideA, (ipiv + j), strideP, A,
                                shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count, scalars,
                                Abyx_norms, (T**)work_workArr);
    }

    // restore original value of A(j,j)
    ROCSOLVER_LAUNCH_KERNEL(restore_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream,
                            diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda, strideA, 1);

    // restore tau
    if(COMPLEX)
        rocsolver_lacgv_template<T>(handle, 1, ipiv, j, 1, strideP, batch_count);
}

return rocblas_status_success;
}
#endif

ROCSOLVER_END_NAMESPACE
