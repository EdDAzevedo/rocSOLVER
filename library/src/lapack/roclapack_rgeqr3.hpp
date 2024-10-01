
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

constexpr int idebug = 0;

#ifndef RGEQR3_BLOCKSIZE
#define RGEQR3_BLOCKSIZE(T) \
    ((sizeof(T) == 4) ? 64 : (sizeof(T) == 8) ? 64 : (sizeof(T) == 16) ? 64 : 32)
#endif

template <typename T, typename I, typename Istride, typename UA, typename UB, typename UC>
static rocblas_status gemm_gemv(rocblas_handle handle,
                                rocblas_operation transA,
                                rocblas_operation transB,
                                I m,
                                I n,
                                I k,

                                T* alpha,

                                UA Amat,
                                Istride shift_Amat,
                                I ldA,
                                Istride stride_Amat,

                                UB Bmat,
                                Istride shift_Bmat,
                                I ldB,
                                Istride stride_Bmat,

                                T* beta,
                                UC Cmat,
                                Istride shift_Cmat,
                                I ldC,
                                Istride stride_Cmat,

                                I batch_count,
                                T** workArr)
{
    //  ------------------------------------
    //  C = beta * C + alpha * op(A) * op(B)
    //  ------------------------------------

    bool const has_work = (m >= 1) && (n >= 1) && (k >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    bool const is_no_transpose_B = (transB == rocblas_operation_none);
    bool const is_transpose_B = (transB == rocblas_operation_transpose);
    bool const is_conj_transpose_B = (transB == rocblas_operation_conjugate_transpose);

    bool const is_no_transpose_A = (transA == rocblas_operation_none);
    bool const is_transpose_A = (transA == rocblas_operation_transpose);
    bool const is_conj_transpose_A = (transA == rocblas_operation_conjugate_transpose);

    I const nrows_A = (is_no_transpose_A) ? m : k;
    I const ncols_A = (is_no_transpose_A) ? k : m;

    I const nrows_B = (is_no_transpose_B) ? k : n;
    I const ncols_B = (is_no_transpose_B) ? n : k;

    if((n == 1) && (!is_conj_transpose_B))
    {
        // -----------------------------------
        // [c1] = beta * [c1] + alpha * [op(A)]  * [b1]
        // [.]           [.]                       [.]
        // [cm]          [cm]                      [bm]
        // -----------------------------------

        auto x = Bmat;
        Istride const offsetx = shift_Bmat;
        I const incx = (is_no_transpose_B) ? 1 : ldB;
        Istride const stridex = stride_Bmat;

        auto y = Cmat;
        Istride const offsety = shift_Cmat;
        I const incy = 1;
        Istride const stridey = stride_Cmat;

        Istride const stride_alpha = 0;
        Istride const stride_beta = 0;
        I const mm = nrows_A;
        I const nn = ncols_A;

        ROCBLAS_CHECK(rocblasCall_gemv(handle, transA, mm, nn, alpha, stride_alpha,

                                       Amat, shift_Amat, ldA, stride_Amat,

                                       x, offsetx, incx, stridex,

                                       beta, stride_beta,

                                       y, offsety, incy, stridey,

                                       batch_count, workArr));
    }
    else if((m == 1) && (!is_conj_transpose_A) && (!is_conj_transpose_B))
    {
        // ---------------------------------------------------------------------------
        // [c1, .., cn] = beta * [c1, ..., cn] + alpha * [a1 ... ak ] * op([B1 |  ... | Bn])
        // ---------------------------------------------------------------------------
        //
        // or take transpose
        //
        // [c1] = beta [c1] + alpha * op2( B ) * [a1]
        // [.]         [.]                       [.]
        // [cn]        [cn]                      [an]
        // ------------------------------------------------------------------------------

        if(idebug >= 1)
        {
            char const c_transA = (is_no_transpose_A) ? 'N'
                : (is_transpose_A)                    ? 'T'
                : (is_conj_transpose_A)               ? 'C'
                                                      : 'X';

            char const c_transB = (is_no_transpose_B) ? 'N'
                : (is_transpose_B)                    ? 'T'
                : (is_conj_transpose_B)               ? 'C'
                                                      : 'X';
            printf("gemm_gemv: m=%d,n=%d,k=%d,transA=%c, transB=%c\n", m, n, k, c_transA, c_transB);
        }
        auto x = Amat;
        Istride offsetx = shift_Amat;
        I incx = (is_no_transpose_A) ? ldA : 1;
        Istride stridex = stride_Amat;

        auto y = Cmat;
        Istride offsety = shift_Cmat;
        I incy = ldC;
        Istride stridey = stride_Cmat;

        Istride const stride_alpha = 0;
        Istride const stride_beta = 0;
        I const mm = nrows_B;
        I const nn = ncols_B;

        rocblas_operation trans_transB
            = (is_no_transpose_B) ? rocblas_operation_transpose : rocblas_operation_none;

        ROCBLAS_CHECK(rocblasCall_gemv(handle, trans_transB, mm, nn, alpha, stride_alpha,

                                       Bmat, shift_Bmat, ldB, stride_Bmat,

                                       x, offsetx, incx, stridex,

                                       beta, stride_beta,

                                       y, offsety, incy, stridey,

                                       batch_count, workArr));
    }
    else
    {
        ROCBLAS_CHECK(rocblasCall_gemm(handle, transA, transB, m, n, k, alpha,

                                       Amat, shift_Amat, ldA, stride_Amat, Bmat, shift_Bmat, ldB,
                                       stride_Bmat,

                                       beta, Cmat, shift_Cmat, ldC, stride_Cmat,

                                       batch_count, workArr));
    }

    return (rocblas_status_success);
}

template <typename T, typename I>
static void rocblasCall_trmm_mem(rocblas_side const side,
                                 I const mm,
                                 I const nn,
                                 I const batch_count,
                                 size_t* size_trmm_byte)
{
    *size_trmm_byte = 2 * sizeof(T*) * std::max(1, batch_count);
}

// -----------------------------------------------
// copy diagonal values
//
// launch as dim(nbx,1,batch_count), dim3(nx,1,1)
// where nbx = ceil( n, nx)
// -----------------------------------------------
template <typename T, typename I, typename Istride>
static __global__ void copy_diagonal_kernel(I const n,

                                            T const* const Tmat,
                                            Istride const shift_Tmat,
                                            I const ldT,
                                            Istride const stride_Tmat,

                                            T* const tau_,
                                            Istride const stride_tau,

                                            I const batch_count)

{
    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const T_p = load_ptr_batch(Tmat, bid, shift_Tmat, stride_Tmat);
        T* const tau_p = tau_ + bid * stride_tau;

        auto Tp = [=](auto i, auto j) -> const T {
            auto const ij = i + j * static_cast<int64_t>(ldT);
            return (T_p[ij]);
        };

        auto tau = [=](auto i) -> T& { return (tau_p[i]); };

        for(auto i = i_start; i < n; i += i_inc)
        {
            tau(i) = Tp(i, i);
        }
    }
}

// -----------------------------------
// copy diagonal entries from T matrix
// into tau array
// -----------------------------------
template <typename T, typename I, typename Istride>
static void copy_diagonal_template(rocblas_handle handle,
                                   I const nn,
                                   T const* const Tmat,
                                   Istride const shift_Tmat,
                                   I const ldT,
                                   Istride const stride_Tmat,
                                   T* const tau_,
                                   Istride const stride_tau,
                                   I const batch_count)
{
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };
    auto min = [](auto x, auto y) { return ((x <= y) ? x : y); };

    auto const max_blocks = 64 * 1000;
    auto const nx = 64;
    auto const nbx = min(max_blocks, ceil(nn, nx));
    auto const nbz = min(max_blocks, batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    copy_diagonal_kernel<<<dim3(nbx, 1, nbz), dim3(nx, 1, 1), 0, stream>>>(
        nn, Tmat, shift_Tmat, ldT, stride_Tmat, tau_, stride_tau, batch_count);
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
static __global__ void geadd_kernel(char const trans,
                                    I const m,
                                    I const n,
                                    T const alpha,
                                    UA AA,
                                    I const shiftA,
                                    I const ldA,
                                    Istride const strideA,
                                    T const beta,
                                    UC CC,
                                    I const shiftC,
                                    I const ldC,
                                    Istride const strideC,
                                    I const batch_count)
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

    bool const is_transpose = (trans == 'T') || (trans == 't');
    bool const is_conj_transpose = (trans == 'C') || (trans == 'c');

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * nx;
    I const i_inc = (nbx * nx);
    I const j_start = hipThreadIdx_y + hipBlockIdx_y * ny;
    I const j_inc = (nby * ny);

    auto const bid_inc = hipGridDim_z;
    auto const bid_start = hipBlockIdx_z;

    T const zero = 0;
    bool const is_beta_zero = (beta == zero);

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const A_ = load_ptr_batch(AA, bid, shiftA, strideA);
        T* const C_ = load_ptr_batch(CC, bid, shiftC, strideC);

        auto A = [=](auto i, auto j) { return (A_[idx2D(i, j, ldA)]); };

        auto C = [=](auto i, auto j) -> T& { return (C_[idx2D(i, j, ldC)]); };

        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = i_start; i < m; i += i_inc)
            {
                auto const aij = (is_transpose) ? A(j, i)
                    : (is_conj_transpose)       ? dconj(A(j, i))
                                                : A(i, j);

                auto const beta_cij = (is_beta_zero) ? zero : beta * C(i, j);
                C(i, j) = beta_cij + alpha * aij;
            }
        }
    }
}

// -----------------------------------------
// geadd() performs matrix addition that is
// similar PxGEADD() in Parallel BLAS library
//
//  C(1:m,1:n) =  beta * C(1:m,1:n) + alpha * op(A)
// -----------------------------------------
template <typename T, typename UA, typename UC, typename I, typename Istride>
static void geadd_template(rocblas_handle handle,
                           char const trans,
                           I const m,
                           I const n,
                           T const alpha,
                           UA AA,
                           Istride const shiftA,
                           I const ldA,
                           Istride const strideA,
                           T const beta,
                           UC CC,
                           Istride const shiftC,
                           I const ldC,
                           Istride const strideC,
                           I const batch_count)
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

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    geadd_kernel<T, UA, UC, I, Istride><<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(
        trans, m, n, alpha, AA, shiftA, ldA, strideA, beta, CC, shiftC, ldC, strideC, batch_count);
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
//
//  Let
//  Y1 = [Y11; Y21; Y31];
//
//  Y1 = [ Y11 ]
//       [ Y21 ]
//       [ Y31 ]
//
//
//  Y2 = [   0 ]
//       [ Y12 ]
//       [ Y22 ]
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
template <typename T, typename I, typename U, typename Istride>
static rocblas_status formT3(rocblas_handle handle,
                             I const m,
                             I const k1,
                             I const k2,

                             U Ymat,
                             Istride const shift_Y1,
                             Istride const shift_Y2,
                             I const ldY,
                             Istride stride_Ymat,

                             T* const Tmat,
                             Istride const shift_T1,
                             Istride const shift_T2,
                             I const ldT,
                             Istride const stride_Tmat,
                             I const batch_count,

                             void* work,
                             I& lwork_bytes)

{
    ROCSOLVER_ENTER("formT3", "m:", m, "k1:", k1, "k2:", k2, "shift_Y1:", shift_Y1, "ldY:", ldY,
                    "bc:", batch_count);

    // 0-based C indexing
    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    // 1-based Fortran indexing
    auto idx2F = [=](auto i, auto j, auto ld) { return (idx2D((i - 1), (j - 1), ld)); };

    auto max = [](auto x, auto y) { return ((x >= y) ? x : y); };

    bool const has_work = (m >= 1) && (k1 >= 1) && (k2 >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    {
        if(work == nullptr)
        {
            return (rocblas_status_invalid_pointer);
        }
    }

    if(idebug >= 1)
    {
        printf("formT3: m=%d, k1=%d, k2=%d, batch_count=%d,lwork_bytes=%d\n", (int)m, (int)k1,
               (int)k2, (int)batch_count, (int)lwork_bytes);
    }

    I total_bytes = 0;
    I remain_bytes = 0;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    std::byte* pfree = (std::byte*)work;

    // W is k1 by k2
    // W  = zeros( size(T1,1), size(T2,2) );

    I const nrows_T1 = k1;
    I const ncols_T1 = k1;

    I const nrows_T2 = k2;
    I const ncols_T2 = k2;

    I const nrows_T3 = nrows_T1;
    I const ncols_T3 = ncols_T2;

    // ---------------------
    // reuse storage T3 be W
    // ---------------------
    I const nrows_W = k1;
    I const ncols_W = k2;

    Istride const shift_T3 = shift_T1 + idx2F(1, k1 + 1, ldT);

    auto Wmat = Tmat;
    Istride const shift_Wmat = shift_T3;
    I const ldW = ldT;
    Istride const stride_Wmat = stride_Tmat;

    // Y1 is m by k1
    // Y2 is (m-k1) by k2

    I const nrows_Y1 = m;
    I const ncols_Y1 = k1;

    I const nrows_Y2 = (m - k1);
    I const ncols_Y2 = k2;

    //
    // m = size(Y1,1);
    // k1 = size(Y1,2);
    // k2 = size(Y2,2);
    // k = k1 + k2;

    I const k = k1 + k2;

    // % -----------------------------------------
    // % Y11 is unit lower triangular size k1 x k1
    // % but Y11 is not used
    // % -----------------------------------------
    // Y11 = Y1(1:k1,1:k1);
    // Y11 = Y11 - triu(Y11) + eye(k1,k1);
    // ---------------------------------------

    Istride const shift_Y11 = shift_Y1 + idx2F(1, 1, ldY);
    I const nrows_Y11 = k1;
    I const ncols_Y11 = k1;

    // % -----------------------------------------
    // % Y12 is unit lower triangular size k2 x k2
    // % -----------------------------------------
    // Y12 = Y2( 1:k2, 1:k2);
    // Y12 = Y12 - triu( Y12 ) + eye( k2,k2);
    // ---------------------------------------
    Istride const shift_Y12 = shift_Y2 + idx2F(1, 1, ldY);
    I const nrows_Y12 = k2;
    I const ncols_Y12 = k2;

    // % -----------------
    // % Y21 is k2 by k1
    // % -----------------
    // Y21 = Y1( (k1+1):(k1 + k2), 1:k1);
    // ----------------------------------
    Istride const shift_Y21 = shift_Y1 + idx2F((k1 + 1), 1, ldY);
    I const nrows_Y21 = k2;
    I const ncols_Y21 = k1;

    // % -----------------
    // % Y31 is (m-k) by k1
    // % -----------------
    // i1 = (k1+k2 + 1);
    // i2 = m;
    // Y31 = Y1( i1:i2, 1:k1);
    // -----------------------
    I i1 = (k1 + k2 + 1);
    I i2 = m;

    Istride const shift_Y31 = shift_Y1 + idx2F(i1, 1, ldY);
    I const nrows_Y31 = (i2 - i1 + 1);
    I const ncols_Y31 = k1;

    assert(nrows_Y31 == (m - k));

    // % ------------------
    // % Y22 is (m-k) by k2
    // % ------------------
    // i2 = size(Y2,1);
    // i1 = (k2+1);
    // Y22 = Y2( i1:i2, 1:k1 );

    i2 = nrows_Y2;
    i1 = (k2 + 1);
    Istride const shift_Y22 = shift_Y2 + idx2F(i1, 1, ldY);
    I const nrows_Y22 = (i2 - i1 + 1);
    I const ncols_Y22 = k2;

    assert(nrows_Y22 == (m - k));

    // % -------------------
    // % (0) first set W =  Y21'
    // % -------------------
    // W = Y21';
    {
        assert(nrows_W == ncols_Y21);
        assert(ncols_W == nrows_Y21);

        char const trans = (rocblas_is_complex<T>) ? 'C' : 'T';
        I const mm = nrows_W;
        I const nn = ncols_W;
        T const alpha = 1;
        T const beta = 0;

        {
            // clang-format off
           geadd_template( handle,
		    trans,
		    mm,
		    nn,
		    alpha,
		    Ymat, shift_Y21, ldY, stride_Ymat,
		    beta,
		    Wmat, shift_Wmat, ldW, stride_Wmat,
		    batch_count );
            // clang-format on
        }
    }

    // % ------------------------
    // % (0) first set W =  Y21'
    // % (1) TRMM   W = (Y21') * Y12
    // % (2) GEMM   W = W + Y31' * Y22
    // % (3) TRMM   W = -T1 * W
    // % (4) TRMM   W = W * T2
    // % ------------------------

    // % --------------------
    // % (1)   W = Y21' * Y12
    // % c_side = 'R';
    // % c_uplo = 'L';
    // % c_trans = 'N';
    // % c_diag = 'U';
    // %  W = trmm( side, uplo, trans, cdiag, mm,nn, alpha, Y12, W );
    // % --------------------
    {
        assert(nrows_Y21 == nrows_Y12);
        assert(ncols_W == ncols_Y12);
        assert(nrows_W == ncols_Y21);

        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_lower;
        rocblas_operation const trans = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_unit;

        auto const mm = nrows_W;
        auto const nn = ncols_W;
        T alpha = 1;
        Istride const stride_alpha = 0;

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);

        size_t const size_workArr = size_trmm_bytes;

        T** const workArr = (T**)pfree;
        pfree += size_trmm_bytes;

        total_bytes += size_trmm_bytes;

        remain_bytes = lwork_bytes - total_bytes;

        {
            assert(remain_bytes >= 0);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm(handle,
			    side, uplo, trans, diag,
			    mm,nn,
			    &alpha, stride_alpha,
			    Ymat, shift_Y12, ldY, stride_Ymat,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr ));
            // clang-format on
        }

        total_bytes = total_bytes - size_trmm_bytes;
        pfree = pfree - size_trmm_bytes;
    }

    // % -----------------------------
    // % (2) GEMM   W = W + Y31' * Y22
    // % transA = 'C';
    // % transB = 'N';
    // % mm = size(W,1);
    // % nn = size(W,2);
    // % kk = size(Y22,1);
    // % -----------------------------
    {
        assert(nrows_W == ncols_Y31);
        assert(nrows_Y31 == nrows_Y22);
        assert(ncols_Y22 == ncols_W);

        rocblas_operation const transA = (rocblas_is_complex<T>)
            ? rocblas_operation_conjugate_transpose
            : rocblas_operation_transpose;
        rocblas_operation const transB = rocblas_operation_none;

        auto const mm = nrows_W;
        auto const nn = ncols_W;
        auto const kk = nrows_Y22;

        T alpha = 1;
        T beta = 1;

        size_t size_gemm = 2 * sizeof(T*) * batch_count;
        T** const workArr = (T**)pfree;
        pfree += size_gemm;
        total_bytes += size_gemm;

        remain_bytes = lwork_bytes - total_bytes;

        {
            assert(remain_bytes >= 0);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        bool const use_gemm_gemv = (mm == 1) || (nn == 1);
        if(use_gemm_gemv)
        {
            // clang-format off
	    ROCBLAS_CHECK( gemm_gemv( handle,
			    transA, transB,
			    mm, nn, kk,
			    &alpha,
			    Ymat, shift_Y31, ldY, stride_Ymat,
			    Ymat, shift_Y22, ldY, stride_Ymat,
			    &beta,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count,
			    workArr ));
            // clang-format on
        }
        else
        {
            // clang-format off
	    ROCBLAS_CHECK( rocblasCall_gemm( handle,
			    transA, transB,
			    mm, nn, kk,
			    &alpha,
			    Ymat, shift_Y31, ldY, stride_Ymat,
			    Ymat, shift_Y22, ldY, stride_Ymat,
			    &beta,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count,
			    workArr ));
            // clang-format on
        }
        total_bytes = total_bytes - size_gemm;
        pfree = pfree - size_gemm;
    }

    // % -----------------------
    // % (3) TRMM    W = -T1 * W
    // % -----------------------
    //
    // side = 'L';
    // uplo = 'U';
    // transA = 'N';
    // cdiag = 'N';
    // mm = size(W,1);
    // nn = size(W,2);
    // alpha = -1;
    //
    // W = trmm( side, uplo, transA, cdiag, mm,nn,alpha, T1, W );

    {
        assert(ncols_T1 == nrows_W);

        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;

        auto const mm = nrows_W;
        auto const nn = ncols_W;

        T alpha = -1;
        Istride const stride_alpha = 0;

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);
        size_t const size_workArr = size_trmm_bytes;

        T** const workArr = (T**)pfree;
        pfree += size_workArr;

        total_bytes += size_workArr;
        remain_bytes = lwork_bytes - total_bytes;

        {
            assert(remain_bytes >= 0);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        {
            // clang-format off
		    ROCBLAS_CHECK( rocblasCall_trmm( handle,
				    side, uplo, transA, diag,
				    mm, nn,
				    &alpha, stride_alpha,
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Wmat, shift_Wmat,  ldW, stride_Wmat,
				    batch_count,
				    workArr ));
            // clang-format on
        }
        total_bytes = total_bytes - size_workArr;
        pfree = pfree - size_workArr;
    }

    // % ---------------------
    // % (4) TRMM   W = W * T2
    // % ---------------------
    // side = 'R';
    // uplo = 'U';
    // transA = 'N';
    // cdiag = 'N';
    // alpha = 1;
    // W = trmm( side, uplo, transA, cdiag, mm,nn,alpha, T2, W );

    {
        assert(ncols_W == nrows_T2);

        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;

        T alpha = 1;
        Istride const stride_alpha = 0;

        I const mm = nrows_W;
        I const nn = ncols_W;

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);
        size_t const size_workArr = size_trmm_bytes;

        T** const workArr = (T**)pfree;
        pfree += size_workArr;
        total_bytes += size_workArr;

        remain_bytes = lwork_bytes - total_bytes;

        {
            assert(remain_bytes >= 1);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        {
            // clang-format off
		    ROCBLAS_CHECK( rocblasCall_trmm( handle,
				    side, uplo, transA, diag,
				    mm, nn,
				    &alpha, stride_alpha,
				    Tmat, shift_T2, ldT, stride_Tmat,
				    Wmat, shift_Wmat, ldW, stride_Wmat,
				    batch_count,
				    workArr ));
            // clang-format on
        }

        total_bytes = total_bytes - size_workArr;
        pfree = pfree - size_workArr;
    }

    return (rocblas_status_success);
}

// ------------------------------
//  Perform C = Q' * C,
//  where Q = eye - Y * T * Y'
//  so Q' = eye - Y * T' * Y
//
//  note Y is lower trapezoidal and has unit diagonal
//
//  C is m by n
//  Y is m by k
//  T is k by k
// -------------------------

template <typename T, typename I, typename UY, typename UC, typename Istride>
static rocblas_status applyQtC(rocblas_handle handle,
                               I const m,
                               I const n,
                               I const k,

                               UY Ymat,
                               Istride const shift_Ymat,
                               I const ldY,
                               Istride const stride_Ymat,

                               T const* const Tmat,
                               Istride const shift_Tmat,
                               I const ldT,
                               Istride const stride_Tmat,

                               UC Cmat,
                               Istride const shift_Cmat,
                               I const ldC,
                               Istride const stride_Cmat,

                               I const batch_count,
                               void* work,
                               I& lwork_bytes)
{
    ROCSOLVER_ENTER("applyQtC", "m:", m, "n:", n, "k:", k, "shift_Ymat:", shift_Ymat, "ldY:", ldY,
                    "bc:", batch_count);
    // 1-based matlab/Fortran indexing
    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto max = [](auto x, auto y) { return ((x >= y) ? x : y); };

    if(idebug >= 1)
    {
        printf("applyQtC: m=%d, n=%d, k=%d,  lwork_bytes=%d\n", (int)m, (int)n, (int)k,
               (int)lwork_bytes);
    }

    I total_bytes = 0;
    I remain_bytes = 0;

    {
        bool const has_work = (m >= 1) && (n >= 1) && (k >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }
    }

    std::byte* pfree = (std::byte*)work;
    {
        if(work == nullptr)
        {
            return (rocblas_status_invalid_pointer);
        }
    }

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // -----------
    // C is m by n
    // Y is m by k
    // T is k by k
    // -----------
    I const nrows_C = m;
    I const ncols_C = n;

    I const nrows_Y = m;
    I const ncols_Y = k;

    I const nrows_T = k;
    I const ncols_T = k;

    //  -------------------
    //  Partition Y and C as
    //   Y = [Y1],    C = [C1]
    //       [Y2]         [C2]
    //
    //
    //   where Y1 is k by k = Y( 1:k,1:k)
    //         Y2 is (m-k) by k = Y( (k+1):m, 1:k)
    //
    // 	C1 is k by n = C(1:k,1:n)
    // 	C2 is (m-k) by n = C( (k+1):m, 1:n )
    //  -------------------

    Istride const shift_C1 = shift_Cmat + idx2F(1, 1, ldC);
    Istride const shift_C2 = shift_Cmat + idx2F((k + 1), 1, ldC);

    Istride const shift_Y1 = shift_Ymat + idx2F(1, 1, ldY);
    Istride const shift_Y2 = shift_Ymat + idx2F((k + 1), 1, ldY);

    I const nrows_C1 = k;
    I const ncols_C1 = n;

    I const nrows_C2 = (m - k);
    I const ncols_C2 = n;

    I const nrows_Y1 = k;
    I const ncols_Y1 = k;

    I const nrows_Y2 = (m - k);
    I const ncols_Y2 = k;

    //   ---------------------------------
    //   [C1] - [Y1] T' * [Y1',  Y2'] * [C1]
    //   [C2]   [Y2]                    [C2]
    //
    //   [C1] - [Y1]  T' * (Y1'*C1 + Y2'*C2)
    //   [C2]   [Y2]
    //
    //   [C1] - [Y1]  T' * W,  where W = Y1'*C1 + Y2'*C2
    //   [C2]   [Y2]
    //
    //   ---------------------------------

    // % --------------------------
    // % (1) W = Y1' * C1, trmm
    // % or
    // % (1a) W = C1,   copy
    // % (1b) W = Y1' * W, trmm
    //
    // % (2) W = W + Y2' * C2, gemm
    // % (3) W = T' * W,   trmm
    // % (4) C2 = C2 - Y2 * W, gemm
    // % (5) W = Y1 * W, trmm
    // % (6) C1 = C1 - W
    // % --------------------------

    // % ------------
    // % (1) W = Y1' * C1;
    // % or
    // % (1a) W = C1,  use copy
    // % (1b) W = (Y1') * W, use trmm
    // % ------------
    //
    // W = C1;
    // side = 'L';
    // transA = 'C';
    // cdiag = 'U';
    // uplo = 'L';
    // alpha = 1;
    // mm = size(C1,1);
    // nn = size(C1,2);
    // W = trmm( side, uplo, transA, cdiag, mm,nn,alpha, Y1, W );

    // allocate storage for Wmat
    //
    I const nrows_W = nrows_C1;
    I const ncols_W = ncols_C1;

    I const ldW = nrows_W;
    Istride const shift_Wmat = 0;
    Istride const stride_Wmat = ldW * ncols_W;

    size_t size_Wmat_bytes = (sizeof(T) * ldW * ncols_W) * batch_count;
    T* Wmat = (T*)pfree;
    pfree += size_Wmat_bytes;

    total_bytes += size_Wmat_bytes;

    remain_bytes = lwork_bytes - total_bytes;

    {
        assert(remain_bytes >= 0);
        if(remain_bytes < 0)
        {
            return (rocblas_status_memory_error);
        }
    }

    {
        // ----------
        // step (1a) W = C1
        // ----------
        char const trans = 'N';
        I const mm = nrows_C1;
        I const nn = ncols_C1;
        T const alpha = 1;
        T const beta = 0;
        Istride const stride_alpha = 0;

        {
            // clang-format off
            geadd_template(handle,
			    trans,
			    mm, nn,
			    alpha,
			    Cmat, shift_C1, ldC, stride_Cmat,
			    beta,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count);
            // clang-format on
        }
    }

    {
        // --------------------------------
        // step (1b) W = (Y1') * W, use trmm
        // --------------------------------

        rocblas_side const side = rocblas_side_left;
        rocblas_operation const trans = (rocblas_is_complex<T>)
            ? rocblas_operation_conjugate_transpose
            : rocblas_operation_transpose;
        rocblas_diagonal const diag = rocblas_diagonal_unit;
        rocblas_fill const uplo = rocblas_fill_lower;

        T alpha = 1;
        Istride const stride_alpha = 0;

        auto const mm = nrows_W;
        auto const nn = ncols_W;

        assert(nrows_W == ncols_Y1);
        assert(nrows_W == nrows_Y1);

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);
        T** const workArr = (T**)pfree;
        pfree += size_trmm_bytes;

        total_bytes += size_trmm_bytes;

        remain_bytes = lwork_bytes - total_bytes;
        {
            assert(remain_bytes >= 0);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm(handle,
			    side, uplo, trans, diag,
			    mm,nn,
			    &alpha, stride_alpha,
			    Ymat, shift_Y1, ldY, stride_Ymat,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr ));
            // clang-format on
        }

        total_bytes = total_bytes - size_trmm_bytes;
        pfree = pfree - size_trmm_bytes;
    }

    // % ----------------
    // % (2) W = W + Y2' * C2;
    // % ----------------
    //
    // transA = 'C';
    // transB = 'N';
    // mm = size(W,1);
    // nn = size(W,2);
    // kk = size(C2,1);
    // alpha = 1;
    // beta = 1;
    // W = gemm( transA, transB, mm,nn,kk, alpha, Y2, C2, beta, W );

    {
        // -------------------------
        // step (2) W = W + Y2' * C2
        // -------------------------

        rocblas_operation const transA = (rocblas_is_complex<T>)
            ? rocblas_operation_conjugate_transpose
            : rocblas_operation_transpose;
        rocblas_operation const transB = rocblas_operation_none;

        auto const mm = nrows_W;
        auto const nn = ncols_W;
        auto const kk = nrows_C2;

        assert(nrows_W == ncols_Y2);
        assert(ncols_W == ncols_C2);
        assert(nrows_Y2 == nrows_C2);

        T alpha = 1;
        T beta = 1;

        size_t const size_workArr = sizeof(T*) * batch_count;
        T** const workArr = (T**)pfree;
        pfree += size_workArr;
        total_bytes += size_workArr;

        remain_bytes = lwork_bytes - total_bytes;
        {
            assert(remain_bytes >= 0);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        bool const use_gemm_gemv = (mm == 1) || (nn == 1);
        if(use_gemm_gemv)
        {
            // clang-format off
            ROCBLAS_CHECK( gemm_gemv( handle,
                            transA, transB,
                            mm, nn, kk,
                            &alpha,
                            Ymat, shift_Y2, ldY, stride_Ymat,
                            Cmat, shift_C2, ldC, stride_Cmat,
                            &beta,
                            Wmat, shift_Wmat, ldW, stride_Wmat,
                            batch_count,
                            workArr ));
            // clang-format on
        }
        else
        {
            // clang-format off
            ROCBLAS_CHECK( rocblasCall_gemm( handle,
                            transA, transB,
                            mm, nn, kk,
                            &alpha,
                            Ymat, shift_Y2, ldY, stride_Ymat,
                            Cmat, shift_C2, ldC, stride_Cmat,
                            &beta,
                            Wmat, shift_Wmat, ldW, stride_Wmat,
                            batch_count,
                            workArr ));
            // clang-format on
        }

        pfree = pfree - size_workArr;
        total_bytes = total_bytes - size_workArr;
    }

    // % ----------
    // % (3) W = T' * W;
    // % ----------
    //
    // side = 'L';
    // uplo = 'U';
    // transA = 'C';
    // cdiag = 'N';
    // mm = size(W,1);
    // nn = size(W,2);
    // alpha = 1;
    // W = trmm( side, uplo, transA, cdiag, mm,nn, alpha, T, W );

    {
        // ----------------------
        // step (3)  W = (T') * W
        // ----------------------

        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const transA = (rocblas_is_complex<T>)
            ? rocblas_operation_conjugate_transpose
            : rocblas_operation_transpose;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;

        auto const mm = nrows_W;
        auto const nn = ncols_W;
        T alpha = 1;
        Istride const stride_alpha = 0;

        assert(nrows_W == ncols_T);
        assert(nrows_W == nrows_T);

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);

        T** const workArr = (T**)pfree;
        pfree += size_trmm_bytes;

        total_bytes += size_trmm_bytes;

        remain_bytes = lwork_bytes - total_bytes;
        {
            assert(remain_bytes >= 0);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm<T>(handle,
			    side, uplo, transA, diag,
			    mm,nn,
			    &alpha, stride_alpha,
			    Tmat, shift_Tmat, ldT, stride_Tmat,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr ));
            // clang-format on
        }

        pfree = pfree - size_trmm_bytes;
        total_bytes = total_bytes - size_trmm_bytes;
    }

    // % ----------------
    // % (4) C2 = C2 - Y2 * W;
    // % ----------------
    //
    // transA = 'N';
    // transB = 'N';
    // mm = size(C2,1);
    // nn = size(C2,2);
    // kk = size(W,1);
    // alpha = -1;
    // beta = 1;
    // C2 = gemm( transA, transB, mm,nn,kk,  alpha, Y2, W, beta, C2 );

    {
        // ----------------------------------
        // step (4)   C2 = C2 - Y2 * W, using gemm
        // ----------------------------------

        rocblas_operation const transA = rocblas_operation_none;
        rocblas_operation const transB = rocblas_operation_none;
        auto const mm = nrows_C2;
        auto const nn = ncols_C2;
        auto const kk = nrows_W;

        assert(nrows_C2 == nrows_Y2);
        assert(ncols_C2 == ncols_W);
        assert(ncols_Y2 == nrows_W);

        T alpha = -1;
        T beta = 1;

        size_t const size_workArr = sizeof(T*) * batch_count;
        T** const workArr = (T**)pfree;
        pfree += size_workArr;

        total_bytes += size_workArr;

        remain_bytes = lwork_bytes - total_bytes;
        {
            assert(remain_bytes >= 0);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        bool const use_gemm_gemv = (mm == 1) || (nn == 1);
        if(use_gemm_gemv)
        {
            // clang-format off
            ROCBLAS_CHECK( gemm_gemv( handle,
                            transA, transB,
                            mm, nn, kk,
                            &alpha,
                            Ymat, shift_Y2,   ldY, stride_Ymat,
                            Wmat, shift_Wmat, ldW, stride_Wmat,
                            &beta,
                            Cmat, shift_C2, ldC, stride_Cmat,
                            batch_count,
                            workArr ));
            // clang-format on
        }
        else
        {
            // clang-format off
            ROCBLAS_CHECK( rocblasCall_gemm( handle,
                            transA, transB,
                            mm, nn, kk,
                            &alpha,
                            Ymat, shift_Y2,   ldY, stride_Ymat,
                            Wmat, shift_Wmat, ldW, stride_Wmat,
                            &beta,
                            Cmat, shift_C2, ldC, stride_Cmat,
                            batch_count,
                            workArr ));
            // clang-format on
        }

        pfree = pfree - size_workArr;
        total_bytes = total_bytes - size_workArr;
    }

    // % ----------
    // % (5) W = Y1 * W, use trmm
    // % ----------
    // side = 'L';
    // uplo = 'L';
    // transA = 'N';
    // cdiag = 'U';
    // alpha = 1;
    // mm = size(W,1);
    // nn = size(W,2);
    // W = trmm( side, uplo, transA, cdiag, mm,nn, alpha, Y1, W );
    {
        // ---------------------
        // step (5)  W = Y1 * W, using trmm
        // ---------------------

        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_lower;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_unit;

        T alpha = 1;
        Istride const stride_alpha = 0;

        I const mm = nrows_W;
        I const nn = ncols_W;

        assert(nrows_W == nrows_Y1);
        assert(nrows_W == ncols_Y1);

        size_t size_trmm_bytes = 0;
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes);

        size_t const size_workArr = size_trmm_bytes;
        T** const workArr = (T**)pfree;
        pfree += size_trmm_bytes;

        total_bytes += size_trmm_bytes;

        remain_bytes = lwork_bytes - total_bytes;
        {
            assert(remain_bytes >= 0);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        {
            // clang-format off
		    ROCBLAS_CHECK( rocblasCall_trmm<T>( handle,
				    side, uplo, transA, diag,
				    mm, nn,
				    &alpha, stride_alpha,
				    Ymat, shift_Y1, ldY, stride_Ymat,
				    Wmat, shift_Wmat,  ldW, stride_Wmat,
				    batch_count,
				    workArr ));
            // clang-format on
        }
        pfree = pfree - size_trmm_bytes;
        total_bytes = total_bytes - size_trmm_bytes;
    }

    //  * -----------
    //  * C1 = C1 - W
    //  * -----------
    {
        char const trans = 'N';
        auto const mm = nrows_W;
        auto const nn = ncols_W;

        assert(nrows_W == nrows_C1);
        assert(ncols_W == ncols_C1);

        T alpha = -1;
        T beta = 1;

        {
            // clang-format off
	      geadd_template( handle,
                    trans,
                    mm,
                    nn,
                    alpha,
                    Wmat, shift_Wmat, ldW, stride_Wmat,
                    beta,
                    Cmat, shift_Cmat, ldC, stride_Cmat,
                    batch_count );
            // clang-format on
        }
    }

    return (rocblas_status_success);
}

template <typename T, typename I>
static void rocsolver_applyQtC_getMemorySize(I const m,
                                             I const n,
                                             I const k,
                                             I const batch_count,
                                             size_t* size_applyQtC)
{
    if(idebug >= 1)
    {
        printf("applyQtC_getMem: begin m=%d,n=%d,k=%d,batch_count=%d,size_applyQtC=%d\n", (int)m,
               (int)n, (int)k, (int)batch_count, (int)*size_applyQtC);
    }

    assert(size_applyQtC != nullptr);

    *size_applyQtC = 0;
    bool const has_work = (m >= 1) && (n >= 1) && (k >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    auto max = [](auto x, auto y) { return ((x >= y) ? x : y); };

    auto const nb = RGEQR3_BLOCKSIZE(T);

    size_t size_trmm_byte = 0;
    {
        rocblas_side const side = rocblas_side_left;
        auto const mm = max(k, nb);
        auto const nn = max(n, nb);
        rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_byte);
    }

    size_t const size_rocblas_byte = 2 * sizeof(T*) * batch_count;
    size_t const size_Wmat_byte = (sizeof(T) * max(k, nb) * max(n, nb)) * batch_count;

    *size_applyQtC = size_trmm_byte + size_rocblas_byte + size_Wmat_byte;

    if(idebug >= 1)
    {
        printf("applyQtC_getMem: end m=%d,n=%d,k=%d,batch_count=%d,size_applyQtC=%d\n", (int)m,
               (int)n, (int)k, (int)batch_count, (int)*size_applyQtC);
    }
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

template <typename T, typename I, typename U, typename Istride>
static rocblas_status rocsolver_rgeqr3_template(rocblas_handle handle,
                                                const I m,
                                                const I n,

                                                U Amat,
                                                const Istride shift_Amat,
                                                const I ldA,
                                                const Istride stride_Amat,

                                                T* const Tmat,
                                                const Istride shift_Tmat,
                                                const I ldT,
                                                const Istride stride_Tmat,

                                                const I batch_count,

                                                void* work,
                                                I& lwork_bytes)
{
    ROCSOLVER_ENTER("rgeqr3", "m:", m, "n:", n, "shift_Amat:", shift_Amat, "lda:", ldA,
                    "bc:", batch_count);

    if(idebug >= 1)
    {
        printf("rgeqr3: m=%d, n=%d, batch_count=%d \n", (int)m, (int)n, (int)batch_count);
    }

    I total_bytes = 0;
    I remain_bytes = lwork_bytes;

    // quick return
    {
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return rocblas_status_success;
        }
    }

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    std::byte* pfree = (std::byte*)work;
    {
        if(work == nullptr)
        {
            return (rocblas_status_invalid_pointer);
        }
    }

    auto idx2F
        = [=](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto max = [](auto x, auto y) { return ((x >= y) ? x : y); };
    auto min = [](auto x, auto y) { return ((x <= y) ? x : y); };
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto sign = [](auto x) { return ((x > 0) ? 1 : (x < 0) ? -1 : 0); };

    I const n_small = 1;
    bool const is_n_small = (n <= n_small);

    if(idebug >= 1)
    {
        printf("rgeqr3:entry: is_n_small=%d, n=%d, m=%d, batch_count=%d\n", (int)is_n_small, (int)n,
               (int)m, (int)batch_count);
    }

    if(n == 1)
    {
        // ------------------------------------------------------
        // generate Householder reflector to work on first column
        // ------------------------------------------------------

        if(idebug >= 1)
        {
            printf("rgeqr3: n == 1, m=%d, batch_count=%d\n", (int)m, (int)batch_count);
        }

        size_t size_work_byte = 0;
        size_t size_norms_byte = 0;
        rocsolver_larfg_getMemorySize<T>(n, batch_count, &size_work_byte, &size_norms_byte);

        // ----------------
        // allocate scratch storage
        // ----------------

        T* const dwork = (T*)pfree;
        pfree += size_work_byte;
        total_bytes += size_work_byte;

        T* const norms = (T*)pfree;
        pfree += size_norms_byte;
        total_bytes += size_work_byte;

        T* tau = Tmat + shift_Tmat;
        Istride const stride_tau = stride_Tmat;
        I const ldtau = 1;
        Istride const shift_tau = 0;

        remain_bytes = lwork_bytes - total_bytes;
        {
            assert(remain_bytes >= 0);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }
        }

        auto alpha = Amat;
        I const shifta = shift_Amat;
        auto x = Amat;
        I const shiftx = shift_Amat + 1;
        I const incx = 1;
        Istride const stridex = stride_Amat;

        {
            rocsolver_larfg_template(handle, m, alpha, shifta, x, shiftx, incx, stridex, tau,
                                     stride_tau, batch_count, dwork, norms);
        }

        pfree = pfree - size_norms_byte;
        pfree = pfree - size_work_byte;

        total_bytes = total_bytes - (size_norms_byte + size_work_byte);
    }
    else if(is_n_small)
    {
        if(idebug >= 0)
        {
            printf("is_n_small: n=%d, m=%d,batch_count=%d\n", (int)n, (int)m, (int)batch_count);
        }

        total_bytes = 0;

        bool constexpr is_batched = true;
        auto A = Amat;
        auto const strideA = stride_Amat;
        I const ishiftA = shift_Amat;
        I const lda = ldA;

        size_t size_scalars = 0;
        size_t size_work_workArr = 0;
        size_t size_Abyx_norms = 0;
        size_t size_diag = 0;

        size_t size_work = 0;
        size_t size_workArr = 0;

        // allocate tau
        size_t size_tau = sizeof(T) * n * batch_count;
        Istride const stride_tau = (sizeof(T) * n);
        I const ldtau = n;
        Istride const shift_tau = 0;

        T* const tau = (T*)pfree;
        pfree += size_tau;

        total_bytes += size_tau;
        // --------------------
        // prepare to use geqr2
        // --------------------
        auto const nn = m;
        auto const kk = n;

        {
            size_t size_scalars_geqr2 = 0;
            size_t size_work_workArr_geqr2 = 0;
            size_t size_Abyx_norms_geqr2 = 0;

            rocsolver_geqr2_getMemorySize<is_batched, T>(m, n, batch_count, &size_scalars_geqr2,
                                                         &size_work_workArr_geqr2,
                                                         &size_Abyx_norms_geqr2, &size_diag);

            size_t size_scalars_larft = 0;
            size_t size_work_larft = 0;
            size_t size_workArr_larft = 0;

            rocsolver_larft_getMemorySize<is_batched, T>(nn, kk, batch_count, &size_scalars_larft,
                                                         &size_work_larft, &size_workArr);

            size_scalars = std::max(size_scalars_geqr2, size_scalars_larft);
            size_Abyx_norms = std::max(size_Abyx_norms_geqr2, size_work_larft);
            size_work_workArr = std::max(size_workArr_larft, size_work_workArr_geqr2);
        }

#if(0)
        rocsolver_geqr2_template<T>(handle, m - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                    (ipiv + j), strideP, batch_count, scalars, work_workArr,
                                    Abyx_norms_trfact, diag_tmptr);

        // apply transformation to the rest of the matrix
        if(j + jb < n)
        {
            // compute block reflector
            rocsolver_larft_template<T>(handle, rocblas_forward_direction, rocblas_column_wise,
                                        m - j, jb, A, shiftA + idx2D(j, j, lda), lda, strideA,
                                        (ipiv + j), strideP, Abyx_norms_trfact, ldw, strideW,
                                        batch_count, scalars, (T*)work_workArr, workArr);

#endif

            T* const scalars = (T*)pfree;
            pfree += size_scalars;

            void* const work_workArr = (void*)pfree;
            pfree += size_work_workArr;

            T* const Abyx_norms = (T*)pfree;
            pfree += size_Abyx_norms;

            T* const diag = (T*)pfree;
            pfree += size_diag;

            T** const workArr = (T**)pfree;
            pfree += size_workArr;

            // -----------------------
            // reuse storage for larft
            // -----------------------
            T* const scalars_larft = (T*)scalars;
            T* const work_larft = (T*)work_workArr;
            T** const workArr_larft = (T**)workArr;

            size_t const total_bytes_geqr2
                = (size_scalars + size_work_workArr + size_Abyx_norms + size_diag);

            total_bytes += total_bytes_geqr2;
            total_bytes += size_workArr;

            remain_bytes = lwork_bytes - total_bytes;
            assert(remain_bytes >= 0);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }

            // -------------------------------------
            // perform factorization of column panel
            // -------------------------------------

            // clang-format off
	    ROCBLAS_CHECK( rocsolver_geqr2_template(
				    handle,
				    m, n,
				    A, ishiftA, lda, strideA,
				    tau, stride_tau,
				    batch_count,
				    scalars,
				    work_workArr,
				    Abyx_norms,
				    diag ));
            // clang-format on

            // -------------
            // form T matrix
            // -------------

#if(0)
            auto const nn = m;
            auto const kk = n;
            rocsolver_larft_getMemorySize<is_batched, T>(nn, kk, batch_count, &size_scalars,
                                                         &size_work, &size_workArr);

            T* const scalars = (T*)pfree;
            pfree += size_scalars;

            T* const work = (T*)pfree;
            pfree += size_work;

            T** const workArr = (T**)pfree;
            pfree += size_workArr;
#endif

            I const ldWm = n;
            size_t const size_Wm = (sizeof(T) * ldWm * n) * batch_count;
            Istride const stride_Wm = (ldWm * n);
            T* const Wm = (T*)pfree;
            pfree += size_Wm;

            total_bytes += size_Wm;
            remain_bytes = lwork_bytes - total_bytes;

            assert(remain_bytes >= 0);
            if(remain_bytes < 0)
            {
                return (rocblas_status_memory_error);
            }

            rocblas_direct const direct = rocblas_forward_direction;
            rocblas_storev const storev = rocblas_column_wise;

            // clang-format off
	     ROCBLAS_CHECK( rocsolver_larft_template( handle,
				     direct,
				     storev,
				     nn,
				     kk,
				     A, ishiftA,ldA,strideA,
				     tau, stride_tau,
				     Wm, ldWm, stride_Wm,
				     batch_count,
				     scalars_larft, work_larft, workArr_larft ));
            // clang-format on

            {
                // -----------
                // copy to Tmat
                // -----------

                char const trans = 'N';
                I const mm = n;
                I const nn = n;
                Istride const shift_Wm = 0;
                T const alpha = 1;
                T const beta = 0;

                // clang-format off
		geadd_template( handle,
				trans,
				mm, nn,
				alpha,
				Wm, shift_Wm, ldWm, stride_Wm,
				beta,
				Tmat, shift_Tmat,ldT, stride_Tmat,
				batch_count );
                // clang-format on
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
            auto const m2 = (m - j1 + 1);

            if(idebug >= 1)
            {
                printf("rgeqr3: m=%d, n1=%d, n2=%d, batch_count=%d\n", (int)m, (int)n1, (int)n2,
                       (int)batch_count);
            }

            auto const k1 = n1;
            auto const k2 = n2;

            // --------------------------------------------
            // [Y1, R1, T1 ] = rgeqr3( A(1:(m), 1:(n1))
            //  where Q1 = eye - Y1 * T1 * Y1'
            // --------------------------------------------

            // ---------------------------------------------
            // Note: Ymat reuses storage in lower trapezoidal
            // part of original Amat
            // ---------------------------------------------
            auto const Ymat = Amat;
            auto const shift_Ymat = shift_Amat;
            auto const stride_Ymat = stride_Amat;
            auto const ldY = ldA;

            auto const shift_Y1 = shift_Ymat + idx2F(1, 1, ldY);
            auto const shift_T1 = shift_Tmat + idx2F(1, 1, ldT);

            auto const nrows_Y1 = m;
            auto const ncols_Y1 = n1;

            auto const nrows_T1 = n1;
            auto const ncols_T1 = n1;

            {
                auto const mm = m;
                auto const nn = n1;

                // clang-format off
              ROCBLAS_CHECK(rocsolver_rgeqr3_template(
                handle,
		mm, nn,
	        Amat, shift_Amat, ldA, stride_Amat,
		Tmat, shift_Tmat, ldT, stride_Tmat,
		batch_count,
		pfree, remain_bytes));
                // clang-format on
            }

            // -----------------------------------------------------
            //
            // compute A(1:m, j1:n) = Q1' * A(1:m, j1:n)
            //
            // where Q1 = eye - Y1 * T1 * Y1',
            // and Y1 is lower trapezoidal with unit diagonal
            //
            // A(1:m,j1:n) = A(1:m,j1:n) - ...
            //   Y1(1:m,1:n1) * (T1(1:n1,1:n1) * (Y1(1:m,1:n1)'*A(1:m,j1:n)));
            // -----------------------------------------------------
            {
                // --------------------------
                // Note: C is alias of A(1:m, j1:n)
                // --------------------------
                auto const Cmat = Amat;
                auto const ldC = ldA;
                auto const shift_Cmat = shift_Amat + idx2F(1, j1, ldA);
                auto const stride_Cmat = stride_Amat;

                auto const mm = m;
                auto const nn = (n - j1 + 1);
                auto const kk = n1;

                {
                    // clang-format off
	      ROCBLAS_CHECK( applyQtC( handle,
				    mm, nn, kk,
				    Ymat, shift_Y1, ldY, stride_Ymat,
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Cmat, shift_Cmat, ldC, stride_Cmat,
				    batch_count,
				    pfree,
				    remain_bytes
				    ) );
                    // clang-format on
                }
            }

            // -----------------------------------------
            // [Y2, R2, T2 ] = rgeqr3( A( j1:m, j1:n ) )
            // -----------------------------------------

            {
                auto const mm = (m - j1 + 1);
                auto const nn = (n - j1 + 1);
                auto const shift_A2 = shift_Amat + idx2F(j1, j1, ldA);
                auto const shift_T2 = shift_Tmat + idx2F(j1, j1, ldT);

                {
                    // clang-format off
                    ROCBLAS_CHECK(rocsolver_rgeqr3_template(
                        handle,
			mm, nn,
			Amat, shift_A2, ldA, stride_Amat,
                        Tmat, shift_T2, ldT, stride_Tmat,
                        batch_count, pfree, remain_bytes));
                    // clang-format on
                }
            }

            // % ------------------------------------------
            // % compute T3 = T(1:n1,j1:n) = -T1(Y1' Y2) T2
            // % ------------------------------------------
            //
            // kk = size(Y1,1) - size(Y2,1);
            // % T3 = -T1 * (Y1' * [ zeros(kk,1); Y2(:)]) * T2;
            // T3 = formT3(  Y1, T1, Y2, T2 );
            // %  T(1:n1,j1:n) = T3;
            // % ------------------------------------------

            {
                // -------------------------------------------------------
                // compute T3 = T(1:n1,j1:n) = -T1(Y1' Y2) T2
                //
                // Note that
                // Y1 is m by n1 unit lower trapezoidal,
                // Y2 is (m-n1) by n2 lower trapezoidal
                // ------------------------------------
                auto const Ymat = Amat;
                Istride const shift_Y1 = shift_Ymat + idx2F(1, 1, ldY);
                Istride const shift_Y2 = shift_Ymat + idx2F(j1, j1, ldY);
                I const ldY = ldA;
                Istride const stride_Ymat = stride_Amat;

                Istride const shift_T1 = shift_Tmat + idx2F(1, 1, ldT);
                Istride const shift_T2 = shift_Tmat + idx2F(j1, j1, ldT);
                Istride const shift_T3 = shift_Tmat + idx2F(1, j1, ldT);

                I const kk1 = n1;
                I const kk2 = n2;
                I const mm = m;

                // -------------------
                // Note: reuse Wmat as T3
                // Let T1 be n1 by n1
                //     T2 be n2 by n2
                // then T3 is n1 by n2
                // -------------------

                {
                    // clang-format off
		    ROCBLAS_CHECK( formT3( handle,
					    mm,  kk1, kk2,
					    Ymat, shift_Y1, shift_Y2, ldY, stride_Ymat,
					    Tmat, shift_T1, shift_T2, ldT, stride_Tmat,
					    batch_count,
					    pfree, remain_bytes ));
                    // clang-format on
                }
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

        return (rocblas_status_success);
    }

    // --------------------------------------
    // estimate the amount of scratch memory
    // required by rgeqr3()
    //
    // This is an over-estimation by
    // it should require  O(  (nb^2) log(nb)  * batch_count)
    // so is still a relatively small amount of storage
    // --------------------------------------
    template <typename T, typename I>
    static void rocsolver_rgeqr3_getMemorySize(I const m, I const n, I const batch_count,
                                               size_t* work_size)
    {
        *work_size = 0;
        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return;
        }

        bool constexpr is_batched = true;
        auto const nb = RGEQR3_BLOCKSIZE(T);

        auto const nlevels = 1 + std::floor(std::log2(static_cast<double>(n)));

        size_t const size_rocblas = (2 * sizeof(T*) * batch_count) * nlevels;
        *work_size += size_rocblas;

        size_t const size_applyQtC = (sizeof(T) * nb * nb) * batch_count * nlevels;
        *work_size += size_applyQtC;

        size_t size_tau = (sizeof(T) * nb) * batch_count;
        *work_size += size_tau;

        {
            size_t size_geqr2 = 0;
            // -----------------
            // scratch space for geqr2
            // -----------------
            size_t size_scalars = 0;
            size_t size_work_workArr = 0;
            size_t size_Abyx_norms = 0;
            size_t size_diag = 0;

            rocsolver_geqr2_getMemorySize<is_batched, T>(
                m, n, batch_count, &size_scalars, &size_work_workArr, &size_Abyx_norms, &size_diag);

            size_geqr2 = (size_scalars + size_work_workArr + size_Abyx_norms + size_diag);
            *work_size += size_geqr2;
        }

        // -----------------------
        // scratch space for larft
        // -----------------------
        size_t size_larft = 0;
        {
            size_t size_scalars = 0;
            size_t size_work = 0;
            size_t size_workArr = 0;

            auto const nn = m;
            auto const kk = nb;
            rocsolver_larft_getMemorySize<is_batched, T>(nn, kk, batch_count, &size_scalars,
                                                         &size_work, &size_workArr);

            size_larft = (size_scalars + size_work + size_workArr);
            *work_size += size_larft;
        }

        size_t size_Wm = (sizeof(T) * nb * nb) * batch_count;
        *work_size += size_Wm;
    }

    // ----------------------------------------------------------
    // perform recursive QR factorization but intended for m >= n
    // tall skinny matrix
    // ----------------------------------------------------------
    template <typename T, typename I, typename UA, typename Istride>
    static rocblas_status rocsolver_rgeqrf_template(rocblas_handle handle, I const m, I const n,

                                                    UA Amat, Istride const shift_Amat, I const ldA,
                                                    Istride const stride_Amat,

                                                    T* tau_, Istride const stride_tau,

                                                    I const batch_count, void* work, I& lwork_bytes)
    {
        ROCSOLVER_ENTER("rgeqrf", "m:", m, "n:", n, "shift_Amat:", shift_Amat, "lda:", ldA,
                        "bc:", batch_count);

        I total_bytes = 0;
        I remain_bytes = 0;

        if(idebug >= 1)
        {
            printf("rgeqrf:m=%d, n=%d, batch_count=%d\n", (int)m, (int)n, (int)batch_count);
        }

        bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
        if(!has_work)
        {
            return (rocblas_status_success);
        }

        if(work == nullptr)
        {
            return (rocblas_status_invalid_pointer);
        }

        hipStream_t stream;
        rocblas_get_stream(handle, &stream);

        // 1-based matlab/Fortran indexing
        auto idx2F
            = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

        auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

        auto max = [](auto x, auto y) { return ((x >= y) ? x : y); };

        auto min = [](auto x, auto y) { return ((x <= y) ? x : y); };

        auto const nb = RGEQR3_BLOCKSIZE(T);

        std::byte* pfree = (std::byte*)work;

        HIP_CHECK(hipMemsetAsync(work, 0, lwork_bytes, stream));

        // -------------
        // allocate Wmat
        // -------------
        I const ldW = nb;
        size_t size_Wmat_bytes = (sizeof(T) * ldW * nb) * batch_count;
        Istride const stride_Wmat = ldW * nb;
        Istride const shift_Wmat = 0;
        T* Wmat = (T*)pfree;
        pfree += size_Wmat_bytes;

        total_bytes += size_Wmat_bytes;
        remain_bytes = lwork_bytes - total_bytes;

        // -------------
        // allocate Tmat
        // -------------
        I const ldT = nb;
        size_t size_Tmat_bytes = (sizeof(T) * ldT * nb) * batch_count;
        Istride const stride_Tmat = ldT * nb;
        Istride const shift_Tmat = 0;

        T* Tmat = (T*)pfree;
        pfree += size_Tmat_bytes;

        total_bytes += size_Tmat_bytes;
        remain_bytes = lwork_bytes - total_bytes;

        assert(remain_bytes >= 0);
        if(remain_bytes < 0)
        {
            return (rocblas_status_memory_error);
        }

        double time_rgeqr3 = 0;
        double time_applyQtC = 0;
        auto tstart = std::chrono::system_clock::now();
        auto tend = std::chrono::system_clock::now();

        for(I j = 1; j <= n; j += nb)
        {
            I const jb = min(n - j + 1, nb);
            I const mm = (m - j + 1);
            I const nn = jb;

            // -------------------------------
            // factorize column panel
            //    [Y,R,T] = rgeqr3(  mm,nn,A(j:m, j:(j+jb-1) )  );
            // -------------------------------

            Istride const shift_Aj = shift_Amat + idx2F(j, j, ldA);

            {
                assert(remain_bytes >= 0);
                if(remain_bytes < 0)
                {
                    return (rocblas_status_memory_error);
                }
            }

            {
                if(idebug >= 1)
                {
                    HIP_CHECK(hipStreamSynchronize(stream));
                    tstart = std::chrono::system_clock::now();
                }

                // clang-format off
                  ROCBLAS_CHECK(rocsolver_rgeqr3_template(
				handle,
				mm, nn,
				Amat, shift_Aj,   ldA, stride_Amat,
				Tmat, shift_Tmat, ldT, stride_Tmat,
				batch_count,
				pfree, remain_bytes));
                // clang-format on

                if(idebug >= 1)
                {
                    HIP_CHECK(hipStreamSynchronize(stream));
                    tend = std::chrono::system_clock::now();
                    std::chrono::duration<double> elapsed_sec = (tend - tstart);
                    time_rgeqr3 += elapsed_sec.count();
                }
            }

            // ----------------------------------------------------
            // copy diagonal entries from T matrix into "tau" array
            // ----------------------------------------------------

            copy_diagonal_template(handle, nn, Tmat, shift_Tmat, ldT, stride_Tmat, tau_, stride_tau,
                                   batch_count);

            // -----------------------------------------------------------
            // update A(j:m,(j+jb):n) = applyQtC( Y, T, A(j:m,(j+jb):n ) );
            // -----------------------------------------------------------

            {
                I const mm = (m - j + 1);
                I const nn = n - (j + jb) + 1;
                I const kk = jb;

                auto Ymat = Amat;
                Istride const shift_Y1 = shift_Amat + idx2F(j, j, ldA);
                I const ldY = ldA;
                Istride const stride_Ymat = stride_Amat;

                Istride const shift_T1 = shift_Tmat + idx2F(1, 1, ldT);

                auto Cmat = Amat;
                Istride const shift_Cmat = shift_Amat + idx2F(j, (j + jb), ldA);
                I const ldC = ldA;
                Istride const stride_Cmat = stride_Amat;

                {
                    if(idebug >= 2)
                    {
                        printf("regqrf: before applyQtC, j=%d, mm=%d,nn=%d,kk=%d\n", (int)j,
                               (int)mm, (int)nn, (int)kk);
                    }

                    if(idebug >= 1)
                    {
                        HIP_CHECK(hipStreamSynchronize(stream));
                        auto tstart = std::chrono::system_clock::now();
                    }
                    // clang-format off
	                ROCBLAS_CHECK( applyQtC( handle,
				    mm, nn, kk,
				    Ymat, shift_Y1, ldY, stride_Ymat,
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Cmat, shift_Cmat, ldC, stride_Cmat,
				    batch_count,
				    pfree,
				    remain_bytes
				    ) );
                    // clang-format on
                    if(idebug >= 1)
                    {
                        HIP_CHECK(hipStreamSynchronize(stream));
                        auto tend = std::chrono::system_clock::now();
                        std::chrono::duration<double> elapsed_sec = (tend - tstart);
                        time_applyQtC += elapsed_sec.count();
                    }
                }
            }

        } // for j

        if(idebug >= 1)
        {
            printf("time_rgeqr3=%le, time_applyQtC=%le\n", time_rgeqr3, time_applyQtC);
        }

        return (rocblas_status_success);
    }

    template <typename T, typename I>
    static void rocsolver_rgeqrf_getMemorySize(I const m, I const n, I const batch_count,
                                               size_t* size_rgeqrf)
    {
        auto const max = [](auto x, auto y) { return ((x >= y) ? x : y); };

        auto const nb = RGEQR3_BLOCKSIZE(T);
        size_t const size_Wmat = (sizeof(T) * nb * max(n, nb)) * batch_count;
        size_t const size_Tmat = (sizeof(T) * nb * nb) * batch_count;
        size_t const size_rocblas = (2 * sizeof(T*)) * batch_count;

        size_t size_rgeqr3 = 0;
        rocsolver_rgeqr3_getMemorySize<T>(m, nb, batch_count, &size_rgeqr3);

        *size_rgeqrf = size_Wmat + size_Tmat + size_rocblas + size_rgeqr3;
    }

    ROCSOLVER_END_NAMESPACE
