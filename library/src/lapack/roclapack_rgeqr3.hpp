
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

#ifndef RGEQR3_BLOCKSIZE
#define RGEQR3_BLOCKSIZE(T) \
    ((sizeof(T) == 4) ? 256 : (sizeof(T) == 8) ? 128 : (sizeof(T) == 16) ? 64 : 32)
#endif

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
// copy diagonal values
//
// launch as dim(nbx,1,batch_count), dim3(nx,1,1)
// where nbx = ceil( n, nx)
// -----------------------------------------------
template <typename T, typename I, typename Istride>
static __global__ void copy_diagonal_kernel(hipStream_t stream,
                                            I const n,

                                            T const* const* Tmat,
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
static void copy_diagonal_template(hipStream_t stream,
                                   I const nn,
                                   T const* const Tmat,
                                   Istride const shift_Tmat,
                                   I const ldT,
                                   Istride const stride_Tmat,
                                   T* const tau_,
                                   Istride const stride_tau,
                                   I batch_count)
{
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };
    auto min = [](auto x, auto y) { return ((x <= y) ? x : y); };

    auto const max_blocks = 64 * 1000;
    auto const nx = 64;
    auto const nbx = ceil(n, nx);
    auto const nbz = min(max_blocks, batch_count);

    copy_diagonal_kernel<T, I, Istride><<<dim3(nbx, 1, nbz), dim3(nx, 1, 1), 0, stream>>>(
        stream, nn, Tmat, shift_Tmat, ldT, stride_Tmat, tau_, stride_tau, batch_count);
}

template <typename T, typename I>
static void
    rocsolver_rgeqr_getMemorySize(I const m, I const n, I const batch_count, size_t* plwork_bytes)
{
    assert(plwork_bytes != nullptr);

    *plwork_bytes = 0;
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        return;
    }

    const size_t nb = RGEQR3_BLOCKSIZE(T);

    size_t const size_T_bytes = sizeof(T) * nb * nb;
    size_t const size_W_bytes = sizeof(T) * nb * nb;

    size_t size_rgeqr3 = 0;
    rocsolver_rgeqr3_getMemorySize(m, nb, batch_count, &size_rgeqr3);

    size_t size_applyQtC = 0;
    rocsolver_applyQtC_getMemorySize(m, n, nb, batch_count, &size_applyQtC);

    size_t const lwork_bytes = (size_T_bytes + size_W_bytes) + size_applyQtC;

    *plwork_bytes = lwork_bytes + size_rgeqr3;
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

        auto C = [=](auto i, auto j) { return (C_[idx2D(i, j, ldC)]); };

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
                             Istride const shift_Y2,
                             I const ldY,
                             Istride stride_Ymat,

                             T* const Tmat,
                             Istride const shift_T1,
                             Istride const shift_T2,
                             I const ldT,
                             Istride const stride_Tmat,

                             T* const Wmat,
                             Istride const shift_Wmat,
                             I const ldW,
                             Istride const stride_Wmat,
                             I const batch_count,

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

    hipStream_t stream;
    if(!is_query)
    {
        rocblas_get_stream(handle, &stream);
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
		    Wmat, shift_Wmat, ldW, stride_Wmat,
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

        size_t size_trmm_bytes = 0;
        ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes));
        size_t const size_workArr = size_trmm_bytes;
        T** const workArr = (T**)pfree;

        total_bytes = max(total_bytes, size_trmm_bytes);

        if(!is_query)
        {
            // clang-format off
	    ROCBLAS_CHECK(rocblasCall_trmm(handle,
			    side, uplo, trans, diag,
			    mm,nn, alpha,

			    Ymat, shift_Y12, ldY, stride_Ymat,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr ));
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
			    Wmat, shift_Wmat, ldW, stride_Wmat,
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
        rocblas_diagonal const diag = rocblas_diagonal_non_unit;

        auto const mm = nrows_W;
        auto const nn = ncols_W;

        T const alpha = -1;

        size_t size_trmm_bytes = 0;
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
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Wmat, shift_Wmat,  ldW, stride_Wmat,
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
        auto const mm = nrows_W;
        auto const nn = ncols_W;

        size_t size_trmm_bytes = 0;
        ROCBLAS_CHECK(rocblasCall_trmm_mem<T>(side, mm, nn, batch_count, &size_trmm_bytes));
        size_t const size_workArr = size_trmm_bytes;
        T** const workArr = (T**)pfree;

        total_bytes = max(total_bytes, size_workArr);

        if(!is_query)
        {
            // clang-format off
		    ROCBLAS_CHECK( rocblasCall_trmm( handle,
				    side, uplo, transA, diag,
				    mm, nn, alpha,
				    Tmat, shift_T2, ldT, stride_Tmat,
				    Wmat, shift_Wmat, ldW, stride_Wmat,
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

                               T* const Wmat,
                               Istride const shift_Wmat,
                               I const ldW,
                               Istride const stride_Wmat,

                               I const batch_count,
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
        T const alpha = 1;

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
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr ));
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
                            Wmat, shift_Wmat, ldW, stride_Wmat,
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
        T const alpha = 1;

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
			    side, uplo, transA, diag,
			    mm,nn, alpha,

			    Tmat, shift_Tmat, ldT, stride_Tmat,
			    Wmat, shift_Wmat, ldW, stride_Wmat,
			    batch_count, workArr ));
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
        total_bytes = max(total_bytes, sizeof(T*) * batch_count);

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

        size_t size_trmm_bytes = 0;
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
				    Wmat, shift_Wmat,  ldW, stride_Wmat,
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

template <typename T, typename I>
static void rocsolver_applyQtC_getMemorySize(I const m,
                                             I const n,
                                             I const k,
                                             I const batch_count,
                                             size_t* size_applyQtC)
{
    assert(size_applyQtC != nullptr);
    *size_applyQtC = 0;

    T* const Ymat = nullptr;
    rocblas_stride const shift_Ymat = 0;
    I const ldY = k;
    rocblas_stride const stride_Ymat = ldY * k;

    T* const Tmat = nullptr;
    rocblas_stride const shift_Tmat = 0;
    I const ldT = k;
    rocblas_stride const stride_Tmat = ldT * k;

    T* const Wmat = nullptr;
    rocblas_stride const shift_Wmat = 0;
    I const ldW = k;
    rocblas_stride const stride_Wmat = ldW * k;

    T* const Cmat = nullptr;
    rocblas_stride const shift_Cmat = 0;
    I const ldC = m;
    rocblas_stride const stride_Cmat = ldC * n;

    void* const work = nullptr;

    rocblas_handle handle;

    // --------------------
    // query work space only
    // --------------------
    I lwork_bytes = -1;

    auto const istat = applyQtC(handle, m, n, k, Ymat, shift_Ymat, ldY, stride_Ymat, Tmat,
                                shift_Tmat, ldT, stride_Tmat, Cmat, shift_Cmat, ldC, stride_Cmat,
                                Wmat, shift_Wmat, ldW, stride_Wmat, batch_count, work, lwork_bytes);

    if(istat == rocblas_status_success)
    {
        *size_applyQtC = lwork_bytes;
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

template <typename T, typename I, typename U, typename Istride, bool COMPLEX = rocblas_is_complex<T>>
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

                                                T* const Wmat,
                                                const Istride shift_Wmat,
                                                const I ldW,
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
    if(!is_query)
    {
        rocblas_get_stream(handle, &stream);
    }

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

    auto sign = [](auto x) { return ((x > 0) ? 1 : (x < 0) ? -1 : 0); };

    if(n == 1)
    {
        // ------------------------------------------------------
        // generate Householder reflector to work on first column
        // ------------------------------------------------------

        size_t size_work_byte = 0;
        size_t size_norms_byte = 0;
        rocsolver_larfg_getMemorySize<T>(n, batch_count, &size_work_byte, &size_norms_byte);

        // ----------------
        // allocate scratch storage
        // ----------------

        T* const dwork = (T*)pfree;
        pfree += size_work_byte;

        T* const norms = (T*)pfree;
        pfree += size_norms_byte;

        size_t const size_tau_byte = sizeof(T) * batch_count;
        Istride const stride_tau = 1;
        T* tau = (T*)tau;
        pfree += size_tau_byte;

        total_bytes = max(total_bytes, size_tau_byte + size_norms_byte + size_work_byte);

        auto alpha = Amat;
        I const shifta = shift_Amat;
        auto x = Amat;
        I const shiftx = shift_Amat + 1;
        I const incx = 1;
        Istride const stridex = stride_Amat;
        I const ldtau = 1;
        Istride const shift_tau = 0;

        if(!is_query)
        {
            rocsolver_larfg_template(handle, m, alpha, shifta, x, shiftx, incx, stridex, tau,
                                     stride_tau, batch_count, dwork, norms);

            I const mm = 1;
            I const nn = 1;
            char const trans = 'N';
            geadd_template(stream, trans, mm, nn, tau, shift_tau, ldtau, Tmat, shift_Tmat, ldT,
                           shift_Tmat, batch_count);
        }

        pfree = pfree - size_tau_byte;
        pfree = pfree - size_norms_byte;
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
                I lwork_n1_bytes = -1;
                // clang-format off
              ROCBLAS_CHECK(rocsolver_rgeqr3_template(
                handle,
		mm, nn,
	        Amat, shift_Amat, ldA, stride_Amat,
		Tmat, shift_Tmat, ldT, stride_Tmat,
                Wmat, shift_Wmat, ldW, stride_Wmat,
		work, lwork_n1_bytes));
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

            if(is_query)
            {
                I lwork_QtC_bytes = 0;
                // clang-format off
	      ROCBLAS_CHECK( applyQtC( handle,
				    mm, nn, kk,
				    Ymat, shift_Y1, ldY, stride_Ymat,
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Cmat, shift_Cmat, ldC, stride_Cmat,
				    Wmat, shift_Wmat, ldW, stride_Wmat,
				    batch_count,
				    work,
				    lwork_QtC_bytes
				    ) );
                // clang-format on
                total_bytes = max(total_bytes, lwork_QtC_bytes);
            }
            else
            {
                // clang-format off
	      ROCBLAS_CHECK( applyQtC( handle,
				    mm, nn, kk,
				    Ymat, shift_Y1, ldY, stride_Ymat,
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Cmat, shift_Cmat, ldC, stride_Cmat,
				    Wmat, shift_Wmat, ldW, stride_Wmat,
				    batch_count,
				    work,
				    lwork_bytes
				    ) );
                // clang-format on
            }

            // -----------------------------------------
            // [Y2, R2, T2 ] = rgeqr3( A( j1:m, j1:n ) )
            // -----------------------------------------

            auto const m2 = (m - j1 + 1);
            auto const n2 = (n - j1 + 1);

            {
                auto const mm = (m - j1 + 1);
                auto const nn = (n - j1 + 1);
                auto const shift_A2 = shift_Amat + idx2F(j1, j1, ldA);
                auto const shift_T2 = shift_Tmat + idx2F(j1, j1, ldT);

                if(is_query)
                {
                    I lwork_n2_bytes = -1;
                    // clang-format off
                    ROCBLAS_CHECK(rocsolver_rgeqr3_template(
                        handle,
			mm, nn,
			Amat, shift_A2, ldA, stride_Amat,
                        Tmat, shift_T2, ldT, stride_Tmat,
			Wmat, shift_Wmat, ldW, stride_Wmat,
                        stride_Wmat, batch_count, work, lwork_n2_bytes));
                    // clang-format on

                    total_bytes = max(total_bytes, lwork_n2_bytes);
                }
                else
                {
                    // clang-format off
                    ROCBLAS_CHECK(rocsolver_rgeqr3_template(
                        handle,
			mm, nn,
			Amat, shift_A2, ldA, stride_Amat,
                        Tmat, shift_T2, ldT, stride_Tmat,
			Wmat, shift_Wmat, ldW, stride_Wmat,
                        stride_Wmat, batch_count, work, lwork_bytes));
                    // clang-format on
                }
            }

            /*
% ------------------------------------------
% compute T3 = T(1:n1,j1:n) = -T1(Y1' Y2) T2
% ------------------------------------------

kk = size(Y1,1) - size(Y2,1);
% T3 = -T1 * (Y1' * [ zeros(kk,1); Y2(:)]) * T2;
T3 = formT3(  Y1, T1, Y2, T2 );
%  T(1:n1,j1:n) = T3;
*/

            {
                // -------------------------------------------------------
                // compute T3 = T(1:n1,j1:n) = -T1(Y1' Y2) T2
                //
                // Note that
                // Y1 is m by n1 unit lower trapezoidal,
                // Y2 is (m-n1) by n2 lower trapezoidal
                // ------------------------------------
                auto const Ymat = Amat;
                auto const shift_Y1 = shift_Ymat + idx2F(1, 1, ldY);
                auto const shift_Y2 = shift_Ymat + idx2F(j1, j1, ldY);
                auto const ldY = ldA;
                auto const stride_Ymat = stride_Amat;

                auto const shift_T1 = shift_Tmat + idx2F(1, 1, ldT);
                auto const shift_T2 = shift_Tmat + idx2F(j1, j1, ldT);
                auto const shift_T3 = shift_Tmat + idx2F(1, j1, ldT);

                auto const kk1 = n1;
                auto const kk2 = n2;
                auto const mm = m;

                // -------------------
                // Note: reuse Wmat as T3
                // Let T1 be n1 by n1
                //     T2 be n2 by n2
                // then T3 is n1 by n2
                // -------------------

                if(is_query)
                {
                    I lwork_formT3_bytes = -1;
                    // clang-format off
		    ROCBLAS_CHECK( formT3( handle,
					    mm,  kk1, kk2,
					    Ymat, shift_Y1, shift_Y2, ldY, stride_Ymat,
					    Tmat, shift_T1, shift_T2, ldT, stride_Tmat,
					    Wmat, shift_Wmat,         ldW, stride_Wmat,
					    batch_count,
					    work, lwork_formT3_bytes ));
                    // clang-format on

                    total_bytes = max(total_bytes, lwork_formT3_bytes);
                }
                else
                {
                    // clang-format off
		    ROCBLAS_CHECK( formT3( handle,
					    mm,  kk1, kk2,
					    Ymat, shift_Y1, shift_Y2, ldY, stride_Ymat,
					    Tmat, shift_T1, shift_T2, ldT, stride_Tmat,
					    Wmat, shift_Wmat,         ldW, stride_Wmat,
					    batch_count,
					    work, lwork_bytes ));
                    // clang-format on
                }

                // ------------
                // copy T3 <- W
                // ------------

                {
                    // clang-format off
                char const trans = 'N';
		auto const mm = n1;
		auto const nn = n2;
		T const alpha = 1;
		T const beta = 0;

	        geadd_template( stream,
                    trans,
                    mm,
                    nn,
                    alpha,
                    Wmat, shift_Wmat, ldW, stride_Wmat,
                    beta,
                    Tmat, shift_T3, ldT, stride_Tmat,
                    batch_count );
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
    }

    if(is_query)
    {
        lwork_bytes = total_bytes;
    }
    return (rocblas_status_success);
}

template <typename T, typename I, typename Istride>
static void rocsolver_rgeqr3_getMemorySize(I const m, I const n, I const batch_count, size_t* work_size)
{
    *work_size = 0;
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    auto min = [](auto x, auto y) { return ((x <= y) ? x : y); };

    auto const nb = min(n, RGEQR3_BLOCKSIZE(T));
    I lwork_bytes = -1;
    void* const work = nullptr;

    T* const Amat = nullptr;
    Istride const shift_Amat = 0;
    I const ldA = m;
    Istride const stride_Amat = ldA * n;

    T* const Tmat = nullptr;
    Istride const shift_Tmat = 0;
    I const ldT = nb;
    Istride const stride_Tmat = ldT * nb;

    T* const Wmat = nullptr;
    Istride const shift_Wmat = 0;
    I const ldW = nb;
    Istride const stride_Wmat = ldW * nb;

    // ---------------------------------
    // call rocsolver_rgeqr3 to query the
    // amount of work space required
    // ---------------------------------

    rocblas_handle handle;

    // clang-format off
    auto const  istat = rocsolver_rgeqr3_template(
        handle,
        m, n,
	Amat, shift_Amat, ldA, stride_Amat,
	Tmat, shift_Tmat, ldT, stride_Tmat,
	Wmat, shift_Wmat, ldW, stride_Wmat,
	batch_count,
	work, lwork_bytes);
    // clang-format on

    if(istat == rocblas_status_success)
    {
        *work_size = lwork_bytes;
    }
}

// ----------------------------------------------------------
// perform recursive QR factorization but intended for m >= n
// tall skinny matrix
// ----------------------------------------------------------
template <typename T, typename I, typename UA, typename Istride>
static rocblas_status roclapack_rgeqrf_template(rocblas_handle handle,
                                                I const m,
                                                I const n,

                                                UA Amat,
                                                Istride const shift_Amat,
                                                I const ldA,
                                                Istride const stride_Amat,

                                                T* tau_,
                                                Istride const stride_tau,
                                                void* work,
                                                I& lwork_bytes)
{
    bool const is_query = (lwork_bytes <= 0);
    if(is_query)
    {
        lwork_bytes = 0;
    }
    I total_bytes = 0;
    I total_applyQtC_bytes = 0;
    I total_rgeqr3_bytes = 0;

    bool const has_work = (m >= 1) && (n >= 1);
    if(has_work)
    {
        if(work == nullptr)
        {
            return (rocblas_status_invalid_pointer);
        }
    }

    // 1-based matlab/Fortran indexing
    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto max = [](auto x, auto y) { return ((x >= y) ? x : y); };

    auto min = [](auto x, auto y) { return ((x <= y) ? x : y); };

    auto const nb = RGEQR3_BLOCKSIZE(T);

    std::byte* pfree = (std::byte*)work;

    // -------------
    // allocate Wmat
    // -------------
    I const ldW = nb;
    size_t size_Wmat_bytes = (sizeof(T) * ldW * nb) * batch_count;
    Istride const stride_Wmat = ldW * nb;
    T* Wmat = (T*)pfree;
    pfree += size_Wmat_bytes;

    total_bytes += size_Wmat_bytes;

    // -------------
    // allocate Tmat
    // -------------
    I const ldT = nb;
    size_t size_Tmat_bytes = (sizeof(T) * ldT * nb) * batch_count;
    Istride const stride_Tmat = ldT * nb;
    T* Tmat = (T*)pfree;
    pfree += size_Wmat_bytes;

    total_bytes += size_Tmat_bytes;

    for(I j = 1; j <= n; j += nb)
    {
        auto const jb = min(n - j + 1, nb);
        auto const mm = (m - j + 1);
        auto const nn = jb;

        // -------------------------------
        // factorize column panel
        //    [Y,R,T] = rgeqr3(  mm,nn,A(j:m, j:(j+jb-1) )  );
        // -------------------------------

        if(is_query)
        {
            I lwork_rgeqr3_bytes = -1;
            // clang-format off
		    auto const  istat = rocsolver_rgeqr3_template(
				handle,
				mm, nn,
				Amat, shift_Amat + idx2F(jstart,jstart,ldA), ldA, stride_Amat,
				Tmat, shift_Tmat, ldT, stride_Tmat,
				Wmat, shift_Wmat, ldW, stride_Wmat,
				batch_count,
				pfree, lwork_geqr3_bytes);
            // clang-format on
            total_rgeqr3_bytes = max(total_rgeqr3_bytes, lwork_geqr3_bytes);
        }
        else
        {
            I lwork_remain_bytes = lwork_bytes - total_bytes;
            if(lwork_remain_bytes <= 0)
            {
                return (rocblas_status_memory_error);
            }

            Istride const shift_A1 = shift_Amat + idx2F(jstart, jstart, ldA);
            // clang-format off
                  auto const  istat = rocsolver_rgeqr3_template(
				handle,
				mm, nn,
				Amat, shift_A1,   ldA, stride_Amat,
				Tmat, shift_Tmat, ldT, stride_Tmat,
				Wmat, shift_Wmat, ldW, stride_Wmat,
				batch_count,
				pfree, lwork_remain_bytes);
            // clang-format on
        }

        // ----------------------------------------------------
        // copy diagonal entries from T matrix into "tau" array
        // ----------------------------------------------------

        copy_diagonal_template(stream, nn, Tmat, shift_Tmat, ldT, stride_Tmat, tau_, stride_tau,
                               batch_count);

        // -----------------------------------------------------------
        // update A(j:m,(j+jb):n) = applyQtC( Y, T, A(j:m,(j+jb):n ) );
        // -----------------------------------------------------------

        {
            auto const mm = (m - j + 1);
            auto const nn = n - (j + jb) + 1;
            auto const kk = jb;

            auto Ymat = Amat;
            auto const shift_Y1 = shift_Amat + idx2F(j, j, ldA);
            auto const ldY = ldA;
            auto const stride_Ymat = stride_Amat;

            auto shift_T1 = shift_Tmat;

            auto Cmat = Amat;
            auto const shift_Cmat = shift_Amat + idx2F(j, (j + jb), ldA);
            auto const ldC = ldA;
            auto const stride_Cmat = stride_Amat;

            if(is_query)
            {
                I lwork_applyQtC_bytes = -1;

                // clang-format off
	                ROCBLAS_CHECK( applyQtC( handle,
				    mm, nn, kk,
				    Ymat, shift_Y1, ldY, stride_Ymat,
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Cmat, shift_Cmat, ldC, stride_Cmat,
				    Wmat, shift_Wmat, ldW, stride_Wmat,
				    batch_count,
				    pfree,
				    lwork_applyQtC_bytes
				    ) );
                // clang-format on

                total_applyQtC_bytes = max(total_applyQtC_bytes, lwork_applyQtC_bytes);
            }
            else
            {
                I lwork_remain_bytes = lwork_bytes - total_bytes;
                if(lwork_remain_bytes <= 0)
                {
                    return (rocblas_status_memory_error);
                }

                // clang-format off
	                ROCBLAS_CHECK( applyQtC( handle,
				    mm, nn, kk,
				    Ymat, shift_Y1, ldY, stride_Ymat,
				    Tmat, shift_T1, ldT, stride_Tmat,
				    Cmat, shift_Cmat, ldC, stride_Cmat,
				    Wmat, shift_Wmat, ldW, stride_Wmat,
				    batch_count,
				    pfree,
				    lwork_remain_bytes
				    ) );
                // clang-format on
            }
        }

    } // for jstart

    if(is_query)
    {
        lwork_bytes = total_bytes + total_applyQtC_bytes + total_rgeqr3_bytes;
    }
    return (rocblas_status_success);
}

ROCSOLVER_END_NAMESPACE
