
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
    *size_trmm_byte = 3 * sizeof(T*) * std::max(1, batch_count);
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

        if((incx == 1) && (incy == 1))
        {
            // ------------
            // special case
            // ------------
            for(I i = i_start; i < n; i += i_inc)
            {
                yp[i] = xp[i];
            }
        }
        else
        {
            for(I i = i_start; i < n; i += i_inc)
            {
                auto const ix = i * incx;
                auto const iy = i * incy;
                yp[iy] = xp[ix];
            }
        }
    }
}

#ifndef RGEQR3_BLOCKSIZE
#define RGEQR3_BLOCKSIZE(T) \
    ((sizeof(T) == 4) ? 256 : (sizeof(T) == 8) ? 128 : (sizeof(T) == 16) ? 64 : 32)
#endif

template <typename T, typename I, typename Istride>
static void rocsolver_rgeqr3_getMemorySize(I const m, I const n, I const batch_count, size_t* work_size)
{
    *work_size = 0;
    bool const has_work = (m >= 1) && (m >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    auto const nb = RGEQR3_BLOCKSIZE(T);
    I lwork_bytes = -1;
    void* work = nullptr;

    T* Amat = nullptr;
    Istride shift_Amat = 0;
    I ldA = m;
    Istride stride_Amat = ldA * n;

    T* Tmat = nullptr;
    Istride shift_Tmat = 0;
    I ldT = nb;
    Istride stride_Tmat = ldT * nb;

    T* Wmat = nullptr;
    Istride shift_Wmat = 0;
    I ldW = nb;
    Istride stride_Wmat = ldW * nb;

    auto istat = rocsolver_rgeqr3_template<BATCHED, STRIDED, T, I, I*, rocblas_stride>(
        m, n, Amat, shift_Amat, ldA, stride_Amat, Tmat, shift_Tmat, ldT, stride_Tmat Wmat,
        shift_Wmat, ldW, stride_Wmat, batch_count, work, lwork_bytes);
    if(istat == rocblas_status_success)
    {
        *work_size = lwork_bytes;
    }

#if(0)
    if(n == 1)
    {
        size_t size_work_byte = 0;
        size_t size_norms_byte = 0;
        rocsolver_larfg_getMemorySize<T>(n, batch_count, &size_work_byte, &size_norms_byte);

        *work_size += (size_work_byte + size_norms_byte);
    }

    auto const n1 = n / 2;
    auto const n2 = n - n1;
    auto const side = rocblas_side_left;
    ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, m, n1, batch_count, &size_trmm_n1))
    ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, m, n2, batch_count, &size_trmm_n2))

    *work_size += std::max(size_trmm_n1, size_trmm_n2);
#endif
}

template <bool BATCHED, typename T, typename I>
static void
    rocsolver_rgeqr_getMemorySize(I const m, I const n, I const batch_count, size_t* plwork_byte)
{
    assert(plwork_byte == nullptr);

    *plwork_byte = 0;
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        return;
    }

    const size_t nb = RGEQR3_BLOCKSIZE(T);

    size_t const size_T_byte = size(T) * nb * nb;
    size_t const size_W_byte = size(T) * nb * nb;

    size_t size_rgeqr3 = 0;
    rocsolver_rgeqr3_getMemorySize(m, nb, batch_count, &size_rgeqr3);

    size_t const lwork_byte = (size_T_byte + size_W_byte)

        * plwork_bye
        = lwork_bye + size_rgeqr3;
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

    bool const is_tranpose = (trans == 'T') || (trans == 't');
    bool const is_conj_transpose = (trans == 'C') || (trans == 'c');

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * nx;
    I const i_inc = (nbx * nx);
    I const j_start = hipThreadIdx_y + hipBlockIdx_y * ny;
    I const j_inc = (nby * ny);

    bool const is_transpose = (trans == 'T' || trans == 't');
    bool const is_conj_transpose = (trans == 'C' || trans == 'c');

    auto const bid_inc = hipGridDim_z;
    auto const bid_start = hipBlockIdx_z;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    T const zero = 0;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const A_ = load_ptr_batch(AA, bid, shiftA, strideA);
        T* const C_ = load_ptr_batch(CC, bid, shiftC, strideC);

        auto A = [=](auto i, auto j) { return (A_[idx2D(i, j, ldA)]); };

        auto C = [=](auto i, auto j) { return (C_[idx2D(i, j, ldC)]); };

        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = i_start; i < m; i += i_inc)
            {
                auto const aij = (is_transpose) ? A(j, i)
                    : (is_conj_transpose)       ? dconj(A(j, i))
                                                : A(i, j);

                auto const cij0 = (beta == zero) ? zero : beta * C(i, j);
                C(i, j) = cij0 + alpha * aij;
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
static void geadd_template(hipStream_t stream,
                           char trans,
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
                             Istride const shift_Y2 I const ldY1,
                             Istride stride_Ymat,

                             T* const Tmat,
                             Istride const shift_T1,
                             Istride const shift_T2,
                             I const ldT,
                             Istride const stride_Tmat,

                             T* const Wmat,
                             Istride const shift_W,
                             I const ldW,
                             Istride const stride_Wmat I const batch_count,

                             void* work,
                             I& lwork_bytes)
{
    // 0-based C indexing
    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    // 1-based Fortran indexing
    auto idx2F = [=](auto i, auto j, auto ld) { return (idx2D((i - 1), (j - 1), ld)); };

    auto max = [](auto x, auto y) { return ((x >= y) ? x : y); };

    // -------------------------------------------
    // check whether it is just query for work sp1ace
    // using LAPACK convention
    // If just query, then just emulate but
    // don't perform any computations
    // -------------------------------------------
    bool const is_query = (lwork_bytes <= 0);
    if(is_query)
    {
        // initialize to defined value
        lwork_bytes = 0;
    }

    size_t total_bytes = 0;

    bool const has_work = (m >= 1) && (k1 >= 1) && (k2 >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    std::byte* pfree = (std::byte*)work;
    if(!is_query)
    {
        if(work == nullptr)
        {
            return (rocblas_status_invalid_pointer);
        }
    }

    // W is k1 by k2
    // W  = zeros( size(T1,1), size(T2,2) );

    auto const nrows_W = k1;
    auto const ncols_W = k2;

    auto const nrows_T1 = k1;
    auto const ncols_T1 = k1;

    auto const nrows_T2 = k2;
    auto const ncols_T2 = k2;

    // Y1 is m by k1
    // Y2 is (m-k1) by k2

    auto const nrows_Y1 = m;
    auto const ncols_Y1 = k1;

    auto const nrows_Y2 = (m - k1);
    auto const ncols_Y2 = k2;

    //
    // m = size(Y1,1);
    // k1 = size(Y1,2);
    // k2 = size(Y2,2);
    // k = k1 + k2;

    auto const k = k1 + k2;

    /*
% -----------------------------------------
% Y11 is unit lower triangular size k1 x k1
% but Y11 is not used
% -----------------------------------------
Y11 = Y1(1:k1,1:k1);
Y11 = Y11 - triu(Y11) + eye(k1,k1);
*/

    auto const shift_Y11 = shift_Y1 + idx2F(1, 1);
    auto const nrows_Y11 = k1;
    auto const ncols_Y11 = k1;

    /*
% -----------------------------------------
% Y12 is unit lower triangular size k2 x k2
% -----------------------------------------
Y12 = Y2( 1:k2, 1:k2);
Y12 = Y12 - triu( Y12 ) + eye( k2,k2);
*/
    auto const shift_Y12 = shift_Y2 + idx2F(1, 1);
    auto const nrows_Y12 = k2;
    auto const ncols_Y12 = k2;

    /*
% -----------------
% Y21 is k2 by k1
% -----------------
Y21 = Y1( (k1+1):(k1 + k2), 1:k1);
*/
    auto const shift_Y21 = shift_Y1 + idx2F((k1 + 1), 1);
    auto const nrows_Y21 = k2;
    auto const ncols_Y21 = k1;

    /*
% -----------------
% Y31 is (m-k) by k1
% -----------------
i1 = (k1+k2 + 1);
i2 = m;
Y31 = Y1( i1:i2, 1:k1);
*/
    auto i1 = (k1 + k2 + 1);
    auto i2 = m;

    auto const shift_Y31 = shift_Y1 + idx2F(i1, 1);
    auto const nrows_Y31 = (i2 - i1 + 1);
    auto const ncols_Y31 = k1;

    assert(nrows_Y31 == (m - k));

    /*
% ------------------
% Y22 is (m-k) by k2
% ------------------
i2 = size(Y2,1);
i1 = (k2+1);
Y22 = Y2( i1:i2, 1:k1 );
*/

    i2 = nrows_Y2;
    i1 = (k2 + 1);
    auto const shift_Y22 = shift_Y2 + idx2F(i1, 1);
    auto const nrows_Y22 = (i2 - i1 + 1);
    auto const ncols_Y22 = k2;

    assert(nrows_Y22 == (m - k));

    /*
% -------------------
% (0) first set W =  Y21'
% -------------------
W = Y21';
*/
    {
        assert(nrows_W == ncols_Y21);
        assert(ncols_W == nrows_Y21);

        char const trans = 'T';
        auto const mm = nrows_W;
        auto const nn = ncols_W;
        T const alpha = 1;
        T const beta = 0;
        // clang-format off
    geadd_template( stream,
		    trans,
		    mm,
		    nn,
		    alpha,
		    Ymat, shift_Y21, ldY, stride_Ymat,
		    beta,
		    Wmat, shift_W, ldW, stride_Wmat,
		    batch_count );
        // clang-format on
    }

    /*
% ------------------------
% (0) first set W =  Y21'
% (1) TRMM   W = (Y21') * Y12
% (2) GEMM   W = W + Y31' * Y22
% (3) TRMM   W = -T1 * W
% (4) TRMM   W = W * T2
% ------------------------
*/

    /*
% --------------------
% (1)   W = Y21' * Y12
% c_side = 'R';
% c_uplo = 'L';
% c_trans = 'N';
% c_diag = 'U';
%  W = trmm( side, uplo, trans, cdiag, mm,nn, alpha, Y12, W );
% --------------------
*/
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
        T const alpha = 1;

        ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_byte));
        size_t const size_workArr = size_trmm_byte;
        T** const workArr = (T**)pfree;

        total_bytes = max(total_bytes, size_trmm_byte);

        if(!is_query)
        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm(handle,
			    side, uplo, trans, diag,
			    mm,nn, alpha,

			    Ymat, shift_Y12, ldY, stride_Ymat,
			    Wmat, shift_W, ldW, stride_Wmat,
			    batch_count, workArr );
        // clang-format on
        }
    }

    /*
% -----------------------------
% (2) GEMM   W = W + Y31' * Y22
% transA = 'C';
% transB = 'N';
% mm = size(W,1);
% nn = size(W,2);
% kk = size(Y22,1);
% -----------------------------
*/
    {
        assert(nrows_W == ncols_Y31);
        assert(nrows_Y31 == nrows_Y22);
        assert(ncols_Y22 = ncols_W);

        rocblas_operation const transA = (rocblas_is_complex<T>)
            ? rocblas_operation_conjugate_transpose
            : rocblas_operation_transpose;
        rocblas_operation const transB = rocblas_operation_none;

        auto const mm = nrows_W;
        auto const nn = ncols_W;
        auto const kk = nrows_Y22;

        T const alpha = 1;
        T const beta = 1;

        T** const workArr = (T**)pfree;
        total_bytes = max(total_bytes, sizeof(T*) * batch_count);

        if(!is_query)
        {
            // clang-format off
	    ROCBLAS_CHECK( rocblasCall_gemm( handle,
			    transA, transB,
			    mm, nn, kk,
			    alpha,
			    Ymat, shift_Y31, ldY, stride_Ymat,
			    Ymat, shift_Y22, ldY, stride_Ymat,
			    beta,
			    Wmat, shift_W, ldW, stride_Wmat,
			    batch_count,
			    workArr ));
            // clang-format on
        }
    }

    /*
% -----------------------
% (3) TRMM    W = -T1 * W
% -----------------------

side = 'L';
uplo = 'U';
transA = 'N';
cdiag = 'N';
mm = size(W,1);
nn = size(W,2);
alpha = -1;

W = trmm( side, uplo, transA, cdiag, mm,nn,alpha, T1, W );
*/
    {
        assert(ncols_T1 == nrows_W);

        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_none;

        auto const mm = nrows_W;
        auto const nn = ncols_W;

        T const alpha = -1;

        ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_byte));
        size_t const size_workArr = size_trmm_byte;
        void* const workArr = (void*)pfree;

        total_bytes = max(total_bytes, size_workArr);

        if(!is_query)
        {
            // clang-format off
		    ROCBLAS_CHECK( rocblasCall_trmm( handle,
				    side, uplo, transA, diag,
				    mm, nn,
				    alpha,
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Wmat, shift_W,  ldW, stride_Wmat,
				    batch_count,
				    workArr ));
            // clang-format on
        }
    }

    /*
% ---------------------
% (4) TRMM   W = W * T2
% ---------------------
side = 'R';
uplo = 'U';
transA = 'N';
cdiag = 'N';
alpha = 1;
W = trmm( side, uplo, transA, cdiag, mm,nn,alpha, T2, W );
*/

    {
        assert(ncols_W == nrows_T2);

        rocblas_side const side = rocblas_side_right;
        rocblas_fill const uplo = rocblas_fill_upper;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;
        T const alpha = 1;

        ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_byte));
        size_t const size_workArr = size_trmm_byte;
        T** const workArr = (T**)pfree;

        total_bytes = max(total_bytes, size_workArr);

        if(!is_query)
        {
            // clang-format off
		    ROCBLAS_CHECK( rocblasCall_trmm( handle,
				    side, uplo, transA, diag,
				    mm, nn, alpha,

				    Tmat, shift_T2, ldT, shift_Tmat,
				    Wmat, shift_W, ldW, shift_Wmat,
				    batch_count,
				    workArr ));
            // clang-format on
        }
    }

    if(is_query)
    {
        lwork_bytes = total_bytes;
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

template <typename T, typename I, typename UY, typename Istride>
static rocblas_status applyQtC(handle,
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

                               T* Wmat,
                               Istride const shift_Wmat,
                               I const ldW,
                               Istride const stride_Wmat,

                               batch_count,
                               void* work,
                               I& lwork_bytes)
{
    // 1-based matlab/Fortran indexing
    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    bool const is_query = (lwork_bytes <= 0);
    if(is_query)
    {
        lwork_bytes = 0;
    }

    size_t total_bytes = 0;

    bool const has_work = (m >= 1) && (n >= 1) && (k >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    std::byte* pfree = (std::byte*)work;
    if(!is_query)
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
    auto const nrows_C = m;
    auto const ncols_C = n;

    auto const nrows_Y = m;
    auto const ncols_Y = k;

    auto const nrows_T = k;
    auto const ncols_T = k;

    /*
 -------------------
 Partition Y and C as
  Y = [Y1],    C = [C1]
      [Y2]         [C2]


  where Y1 is k by k = Y( 1:k,1:k)
        Y2 is (m-k) by k = Y( (k+1):m, 1:k)

	C1 is k by n = C(1:k,1:n)
	C2 is (m-k) by n = C( (k+1):m, 1:n )
 -------------------
*/

    auto const shift_C1 = shift_Cmat + idx2F(1, 1);
    auto const shift_C2 = shift_Cmat + idx2F((k + 1), 1);

    auto const shift_Y1 = shift_Ymat + idx2F(1, 1);
    auto const shift_Y2 = shift_Ymat + idx2F((k + 1), 1);

    auto const nrows_C1 = k;
    auto const ncols_C1 = n;

    auto const nrows_C2 = (m - k);
    auto const ncols_C2 = n;

    auto const nrows_Y1 = k;
    auto const ncols_Y1 = k;

    auto const nrows_Y2 = (m - k);
    auto const ncols_Y2 = k;

    /*
  ---------------------------------
  [C1] - [Y1] T' * [Y1',  Y2'] * [C1]
  [C2]   [Y2]                    [C2]

  [C1] - [Y1]  T' * (Y1'*C1 + Y2'*C2)
  [C2]   [Y2]

  [C1] - [Y1]  T' * W,  where W = Y1'*C1 + Y2'*C2
  [C2]   [Y2]

  ---------------------------------
*/

    /*
% --------------------------
% (1) W = Y1' * C1, trmm
% or
% (1a) W = C1,   copy
% (1b) W = Y1' * W, trmm

% (2) W = W + Y2' * C2, gemm
% (3) W = T' * W,   trmm
% (4) C2 = C2 - Y2 * W, gemm
% (5) W = Y1 * W, trmm
% (6) C1 = C1 - W
% --------------------------
*/

    /*
% ------------
% (1) W = Y1' * C1;
% or
% (1a) W = C1,  use copy
% (1b) W = (Y1') * W, use trmm
% ------------

W = C1;
side = 'L';
transA = 'C';
cdiag = 'U';
uplo = 'L';
alpha = 1;
mm = size(C1,1);
nn = size(C1,2);
W = trmm( side, uplo, transA, cdiag, mm,nn,alpha, Y1, W );
*/

    auto const nrows_W = nrows_C1;
    auto const ncols_W = ncols_C1;
    {
        // ----------
        // step (1a) W = C1
        // ----------
        char const trans = 'N';
        auto const mm = nrows_C1;
        auto const nn = ncols_C1;

        if(!is_query)
        {
            geadd_template(stream, trans, mm, nn, alpha, Cmat, shift_C1, ldC, stride_Cmat, Wmat,
                           shift_Wmat, ldW, stride_Wmat, batch_count);
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

        T const alpha = 1;

        auto const mm = nrows_C1;
        auto const nn = ncols_C1;

        assert(nrows_W == ncols_Y1);
        assert(nrows_W == nrows_Y1);

        size_t size_trmm_bytes = 0;
        ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes));
        T** const workArr = (T**)pfree;

        total_bytes = max(total_bytes, size_trmm_bytes);

        if(!is_query)
        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm(handle,
			    side, uplo, trans, diag,
			    mm,nn, alpha,

			    Ymat, shift_Y1, ldY, stride_Ymat,
			    Wmat, shift_W, ldW, stride_Wmat,
			    batch_count, workArr );
        // clang-format on
        }
    }

    /*
% ----------------
% (2) W = W + Y2' * C2;
% ----------------

transA = 'C';
transB = 'N';
mm = size(W,1);
nn = size(W,2);
kk = size(C2,1);
alpha = 1;
beta = 1;
W = gemm( transA, transB, mm,nn,kk, alpha, Y2, C2, beta, W );
*/

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

        T const alpha = 1;
        T const beta = 1;

        T** const workArr = (T**)pfree;
        size_t const size_workArr = sizeof(T*) * batch_count;
        total_bytes = max(total_bytes, size_workArr);

        if(!is_query)
        {
            // clang-format off
            ROCBLAS_CHECK( rocblasCall_gemm( handle,
                            transA, transB,
                            mm, nn, kk,
                            alpha,
                            Ymat, shift_Y2, ldY, stride_Ymat,
                            Cmat, shift_C2, ldC, stride_Cmat,
                            beta,
                            Wmat, shift_W, ldW, stride_Wmat,
                            batch_count,
                            workArr ));
            // clang-format on
        }
    }

    /*
% ----------
% (3) W = T' * W;
% ----------

side = 'L';
uplo = 'U';
transA = 'C';
cdiag = 'N';
mm = size(W,1);
nn = size(W,2);
alpha = 1;
W = trmm( side, uplo, transA, cdiag, mm,nn, alpha, T, W );
*/

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

        assert(nrows_W == ncols_T);
        assert(nrows_W == nrows_T);

        size_t size_trmm_bytes = 0;
        ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes));
        T** const workArr = (T**)pfree;

        total_bytes = max(total_bytes, size_trmm_bytes);

        if(!is_query)
        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm(handle,
			    side, uplo, trans, diag,
			    mm,nn, alpha,

			    Tmat, shift_Tmat, ldT, stride_Tmat,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr );
        // clang-format on
        }
    }

    /*
% ----------------
% (4) C2 = C2 - Y2 * W;
% ----------------

transA = 'N';
transB = 'N';
mm = size(C2,1);
nn = size(C2,2);
kk = size(W,1);
alpha = -1;
beta = 1;
C2 = gemm( transA, transB, mm,nn,kk,  alpha, Y2, W, beta, C2 );
*/

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

        T const alpha = -1;
        T const beta = 1;

        T** const workArr = (T**)pfree;
        total_bytes = max(total_bytes, sizeof(T*) batch_count);

        if(!is_query)
        {
            // clang-format off
            ROCBLAS_CHECK( rocblasCall_gemm( handle,
                            transA, transB,
                            mm, nn, kk,
                            alpha,
                            Ymat, shift_Y2,   ldY, stride_Ymat,
                            Wmat, shift_Wmat, ldW, stride_Wmat,
                            beta,
                            Cmat, shift_C2, ldC, stride_Cmat,
                            batch_count,
                            workArr ));
            // clang-format on
        }
    }

    /*
% ----------
% (5) W = Y1 * W, use trmm
% ----------
side = 'L';
uplo = 'L';
transA = 'N';
cdiag = 'U';
alpha = 1;
mm = size(W,1);
nn = size(W,2);
W = trmm( side, uplo, transA, cdiag, mm,nn, alpha, Y1, W );
*/
    {
        // ---------------------
        // step (5)  W = Y1 * W, using trmm
        // ---------------------

        rocblas_side const side = rocblas_side_left;
        rocblas_fill const uplo = rocblas_fill_lower;
        rocblas_operation const transA = rocblas_operation_none;
        rocblas_diagonal const diag = rocblas_diagonal_unit;
        T const alpha = 1;

        auto const mm = nrows_W;
        auto const nn = ncols_W;

        assert(nrows_W == nrows_Y1);
        assert(nrows_W == ncols_Y1);

        ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes));
        size_t const size_workArr = size_trmm_bytes;
        void* const workArr = (void*)pfree;

        total_bytes = max(total_bytes, size_workArr);

        if(!is_query)
        {
            // clang-format off
		    ROCBLAS_CHECK( rocblasCall_trmm( handle,
				    side, uplo, transA, diag,
				    mm, nn,
				    alpha,
				    Ymat, shift_Y1, ldY, stride_Ymat,
				    Wmat, shift_W,  ldW, stride_Wmat,
				    batch_count,
				    workArr ));
            // clang-format on
        }
    }

    /*
 * -----------
 * C1 = C1 - W
 * -----------
 */
    {
        char const trans = 'N';
        auto const mm = nrows_W;
        auto const nn = ncols_W;

        assert(nrows_W == nrows_C1);
        assert(ncols_W == ncols_C1);

        T const alpha = -1;
        T const beta = 1;

        if(!is_query)
        {
            // clang-format off
	      geadd_template( stream,
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

    if(is_query)
    {
        lwork_bytes = total_bytes;
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

                                                U Amat,
                                                const Istride shift_Amat,
                                                const I ldA,
                                                const Istride stride_Amat,

                                                T* const Tmat,
                                                const Istride shift_Tmat,
                                                const I ldt,
                                                const Istride stride_Tmat,

                                                T* const Wmat,
                                                const Istride shift_Wmat,
                                                const I ldw,
                                                const Istride stride_Wmat,

                                                const I batch_count,

                                                void* work,
                                                I& lwork_bytes)

{
    ROCSOLVER_ENTER("rgeqr3", "m:", m, "n:", n, "shift_Amat:", shift_Amat, "lda:", ldA,
                    "bc:", batch_count);

    bool const is_query = (lwork_bytes <= 0);
    if(is_query)
    {
        lwork_bytes = 0;
    }

    size_t total_bytes = 0;

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
    if(!is_query)
    {
        if(work == nullptr)
        {
            return (rocblas_status_invalid_pointer);
        }
    }

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto idx2F = [=](auto i, auto j, auto ld) { return (idx2D((i - 1), (j - 1), ld)); };

    auto max = [](auto x, auto y) { return ((x >= y) ? x : y); };
    auto min = [](auto x, auto y) { return ((x <= y) ? x : y); };
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    if(n == 1)
    {
        // ------------------------------------------------------
        // generate Householder reflector to work on first column
        // ------------------------------------------------------
        const I j = 0;
        Istride const stride_Tmat = (ldt * n);

        size_t size_work_byte = 0;
        size_t size_norms_byte = 0;
        rocsolver_larfg_getMemorySize<T>(n, batch_count, &size_work_byte, &size_norms_byte);

        // ----------------
        // allocate scratch storage
        // ----------------

        T* const work = (T*)pfree;
        pfree += size_work_byte;

        T* const norms = (T*)pfree;
        pfree += size_norm_byte;

        total_bytes = max(total_bytes, size_norm_byte + size_work_byte);

        auto alpha = A;
        auto shifta = shift_Amat;
        auto x = A;
        auto shiftx = shift_Amat + 1;
        rocblas_int incx = 1;
        auto stridex = stride_Amat;
        T* tau = Tmat + shift_Tmat;
        auto strideP = stride_Tmat;

        if(!is_query)
        {
            rocsolver_larfg_template(handle, m, alpha, shifta, x, shiftx, incx, stridex, tau,
                                     strideP, batch_count, work, norms);
        }

        pfree = pfree - size_norm_byte;
        pfree = pfree - size_work_byte;
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

            if(is_query)
            {
                I lwork_n1 = -1;
                // clang-format off
              ROCBLAS_CHECK(rocsolver_rgeqr3_template(
                handle,
		mm, nn,
	        Amat, shift_Amat, ldA, stride_Amat,
		Tmat, shift_Tmat, ldT, stride_Tmat,
                Wmat, shift_Wmat, ldW, stride_Wmat,
		work, lwork_n1));
                // clang-format on

                total_bytes = max(total_bytes, lwork_n1_bytes);
            }
            else
            {
                // clang-format off
              ROCBLAS_CHECK(rocsolver_rgeqr3_template(
                handle,
		mm, nn,
	        Amat, shift_Amat, ldA, stride_Amat,
		Tmat, shift_Tmat, ldT, stride_Tmat,
                Wmat, shift_Wmat, ldW, stride_Wmat,
		work, lwork_bytes));
                // clang-format on
            }
        }

        /*
  % -----------------------------------------
  % compute A(1:m, j1:n) = Q1' * A(1:m, j1:n)
  % -----------------------------------------
  % A(1:m, j1:n) = (eye - Y1*T1*Y1') * A(1:m, j1:n)

  use_applyQtC = 1;
  if (use_applyQtC),
	A(1:m, j1:n) = applyQtC(   Y1, T1, A(1:m,j1:n) );
  else
    A(1:m,j1:n) = A(1:m,j1:n) - ...
	  Y1(1:m,1:n1) * (T1(1:n1,1:n1) * (Y1(1:m,1:n1)'*A(1:m,j1:n)));
  end;

*/

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

                T* const tmptr = (T*)pfree;
                pfree += size_tmptr_byte;

                T* const workArr = (T*)pfree;
                pfree += size_workArr_byte;

                total_bytes = max(total_bytes, size_tmptr_byte + size_workArr_byte);

                // -----------------------------------------------
                // call larfb to apply block Householder reflector
                // -----------------------------------------------

                const rocblas_side side = rocblas_side_left;
                const rocblas_operation trans = rocblas_operation_conjugate_transpose;
                const rocblas_direct direct = rocblas_forward_direction;
                const rocblas_storev storev = rocblas_column_wise;

                auto const Ymat = Amat;
                auto const shift_Ymat = shift_Amat;
                auto const ldY = ldA;
                auto const stride_Ymat = stride_Amat;

                if(!is_query)
                {
                    ROCBLAS_CHECK(rocsolver_larfb_template<BATCHED, STRIDED, T>(
                        handle, side, trans, direct, storev, mm, nn, kk, Ymat, shift_Y, ldY,
                        stride_Ymat, Tmat, shift_T, ldT, stride_Tmat, Amat,
                        shift_A + idx2D(0, n1, lda), lda, stride_Amat, batch_count, tmptr, workArr));
                }
                // ------------------------
                // release memory for larfb
                // ------------------------

                pfree = pfree - size_tmptr_byte;
                pfree = pfree - size_workArr_byte;
            }

            // -----------------------------------------
            // [Y2, R2, T2 ] = rgeqr3( A( n1:m, n1:n ) )
            // -----------------------------------------

            {
                auto const mm = (m - j1 + 1);
                auto const nn = (n - j1 + 1);
                auto const A2_offset = idx2D((j1 - 1), (j1 - 1), lda);
                auto const T2_offset = idx2D(k1, k1, ldt);

                if(is_query)
                {
                    I lwork_n2_bytes = -1;
                    ROCBLAS_CHECK(rocsolver_rgeqr3_template<BATCHED, STRIDED, T>(
                        handle, mm, nn, batch_count, A, shift_Amat + A2_offset, lda, stride_Amat,
                        Tmat, shift_Tmat + T2_offset, ldt, stride_Tmat, Wmat, shift_Wmat, ldw,
                        stride_Wmat, work, lwork_n2_bytes));

                    total_bytes = max(total_bytes, lwork_n2_bytes);
                }
                else
                {
                    ROCBLAS_CHECK(rocsolver_rgeqr3_template<BATCHED, STRIDED, T>(
                        handle, mm, nn, batch_count, A, shift_Amat + A2_offset, lda, stride_Amat,
                        Tmat, shift_Tmat + T2_offset, ldt, stride_Tmat, Wmat, shift_Wmat, ldw,
                        stride_Wmat, pfree, lwork_bytes));
                }
            }

            {
                // -------------------------------------------------------
                // compute T3 = T(0:(n1-1), n1:n ) = -T1 * (Y1' * Y2) * T2
                //
                // Note that
                // Y1 is m by n1 unit lower trapezoidal,
                // Y2 is (m-n1) by n2 lower trapezoidal
                // ------------------------------------
                auto Ymat = Amat;
                auto const shiftY = shift_Amat;
                auto const ldy = lda;
                auto const strideY = stride_Amat;

                if(is_query)
                {
                    I lwork_formT3_bytes = -1;
                    ROCBLAS_CHECK(formT3<T, I, U, Istride>(handle, m, n1, n2, batch_count, Y, shiftY,
                                                           ldy, strideY, Tmat, shift_Tmat, ldt,
                                                           stride_Tmat, work, lwork_formT3_bytes));
                    total_bytes = max(total_bytes, lwork_formT3_bytes);
                }
                else
                {
                    ROCBLAS_CHECK(formT3<T, I, U, Istride>(handle, m, n1, n2, batch_count, Y,
                                                           shiftY, ldy, strideY, Tmat, shift_Tmat,
                                                           ldt, stride_Tmat, pfree, lwork_bytes));
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
    }

    if(is_query)
    {
        lwork_bytes = total_bytes;
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
