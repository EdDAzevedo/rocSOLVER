
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

template <bool BATCHED, typename T>
static void rocsolver_rgeqr3_getMemorySize(const rocblas_int m,
                                           const rocblas_int n,
                                           const rocblas_int batch_count,
                                           size_t* size_T,
                                           size_t* size_W)
{
    *size_T = 0;
    *size_W = 0;
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        return;
    }

    const size_t nb = GEQxF_BLOCKSIZE;
    *size_T = sizeof(T) * nb * nb * std::max(1, batch_count);
    *size_W = sizeof(T) * nb * nb * std::max(1, batch_count);
}

static double dconj(double x)
{
    return (x);
};
static float dconj(float x)
{
    return (x);
};
static std::complex<float> dconj(std::complex<float> x)
{
    return (std::conj(x));
};
static std::complex<double> dconj(std::complex<double> x)
{
    return (std::conj(x));
};
static rocblas_complex_num<float> dconj(rocblas_complex_num<float> x)
{
    return (conj(x));
};

static rocblas_complex_num<double> dconj(rocblas_complex_num<double> x)
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
static void geadd_kernel(char trans,
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

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * nx;
    I const i_inc = (nbx * nx);
    I const j_start = hipThreadIdx_y + hipBlockIdx_y* ny I const j_inc = (nby * ny);

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
                           I const shiftA,
                           I const ldA,
                           Istride const strideA,
                           T const beta,
                           UC CC,
                           I const shiftC,
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
template <typename T, typename U, typename I, typename Istride>
static rocblas_status formT3(I const m,
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
                             Istride const strideT)
{
    Istride const T1_offset = idx2D(0, 0, ldT);
    Istride const T2_offset = idx2D(k1, k1, ldT);

    T* const Wmat = Tmat;
    Istride const shiftW = shiftT + idx2D(0, k1, ldT);
    auto const ldW = ldT auto const strideW = strideT;

    auto const k = k1 + k2;

    Istride const Y11_offset = idx2D(0, 0, ldY);
    Istride const Y21_offset = idx2D(k1, 0, ldY);
    Istride const Y31_offset = idx2D(k, 0, ldy);

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

    auto const Y11_offset = idx2D(0, 0, ldY1);
    auto const Y21_offset = idx2D(k1, 0, ldY1);
    auto const Y31_offset = idx2D(k1 + k2, 0, ldY1);

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
        I const mm = ncols_Y21;
        I const nn = nrows_Y21;
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

        HIP_CHECK(rocblasCall_trmm<T>(handle, side, uplo, transA, diag, mm, nn, &alpha,
                                      stride_alpha, Y, shiftY + Y21_offset, ldY2, strideY, Wmat,
                                      shiftW, ldW, strideW, batch_count));
    }

    {
        // -----------------------------
        // (2) GEMM   W = W + Y31' * Y22
        // -----------------------------

        rocblas_operation const trans_A = rocblas_operation_conjugate_transpose;
        rocblas_operation const trans_B = rocblas

            I const mm
            = nrow_W;
        I const nn = ncol_W;
        I const kk = nrow_Y22;

        T alpha = 1;
        T beta = 1;

        T** work = nullptr;
   HIP_CHECK( rocblasCall_gemm<T>( handle, 
			   trans_A, trans_B, mm,nn,kk,
			   &alpha,  
			   Y, shiftY + Y31_offset, ldY, strideY,
			   Y, shiftY + Y22_offset, ldY, strideY,
			   &beta,
			   Wmat, shiftW, ldW, strideW,
			   batch_count,  work );
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
        Istride const stride_alpha = 0;

        HIP_CHECK(rocblasCall_trmm<T>(handle, side, uplo, transA, diag, mm, nn, &alpha,
                                      stride_alpha, Tmat, shiftT + T1_offset, ldT, strideT, Wmat,
                                      shiftW, ldW, strideW, batch_count));
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

        HIP_CHECK(rocblasCall_trmm<T>(handle, side, uplo, transA, diag, mm, nn, &alpha,
                                      stride_alpha, Tmat, shiftT + T2_offset, ldT, strideT, Wmat,
                                      shiftW, ldW, strideW, batch_count));
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

template <typename T, typename U, typename I, typename Istride, bool COMPLEX = rocblas_is_complex<T>>
static rocblas_status rocsolver_rgeqr3_template(rocblas_handle handle,
                                                const I m,
                                                const I n,
                                                U A,
                                                const Istride shiftA,
                                                const I lda,
                                                const Istride strideA,
                                                const I batch_count,
                                                T* const Tmat,
                                                I const ldT,
                                                T* const Wmat,
                                                I const ldW,
                                                T* work_workArr,
                                                T* Abyx_norms)
{
    ROCSOLVER_ENTER("rgeqr3", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto min = [](auto x, auto y) { return ((x < y) ? x : y); };

    if(n == 1)
    {
        // ------------------------------------------------------
        // generate Householder reflector to work on first column
        // ------------------------------------------------------
        const I j = 0;
        Istride const strideT = (ldT * n);

        rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(j, j, lda), A,
                                 shiftA + idx2D(min(j + 1, m - 1), j, lda), 1, strideA, Tmat,
                                 strideT, batch_count, (T*)work_workArr, Abyx_norms);
    }
    else
    {
        auto const n1 = n / 2;
        auto const n2 = n - n1;

        // --------------------------------------------
        // [Y1, R1, T1 ] = rgeqr3( A(0:(m-1), 0:(n1-1))
        // --------------------------------------------

        {
            T* const T1 = Tmat;
            HIP_CHECK(rocsolver_rgeqr3_template(handle, m, k1, A, shiftA, lda, strideA, batch_count,
                                                work_workArr, Abyx_norms, T1, ldT, Wmat, ldW));
        }

        // -----------------------------------------------------
        // compute  A(0:(m-1), n1:n ) = Q1' * A( 0:(m-1), n1:n )
        // -----------------------------------------------------

        {
            auto const Y1_offset = idx2D(0, 0, ldA);
            Istride const shiftY = idx2D(0, n1, ldA);
            HIP_CHECK(applyQtC(stream, m, n2, n1, A + shiftA + Y1_offset, lda, strideA, // Y1
                               Tmat, ldT, // T1
                               A + shiftA + shiftA2, lda, strideA));
        }

        // -----------------------------------------
        // [Y2, R2, T2 ] = rgeqr3( A( n1:m, n1:n ) )
        // -----------------------------------------

        T* const T2 = Tmat + idx2D(n1, n1, ldT);

        HIP_CHECK(rocsolver_rgeqr3_template(handle, m, n2, A, shiftA + shiftA2, lda, strideA,
                                            batch_count, work_workArr, Abyx_norms, T2, ldT, Wmat,
                                            ldW));

        // -------------------------------------------------------
        // compute T3 = T(0:(n1-1), n1:n ) = -T1 * (Y1' * Y2) * T2
        // -------------------------------------------------------

        // ------------------------------------
        // Note that
        // Y1 is m by n1 unit lower trapezoidal,
        // Y2 is (m-n1) by n2 lower trapezoidal
        // ------------------------------------
        T* const T3 = Tmat + idx2D(0, n1, ldT);
        HIP_CHECK( formT3( m, n1, n2,  
				    A, shiftA + Y1_offset, lda, strideA, // Y1
				    A, shiftA + Y2_offset, lda, strideA, // Y2
				    T1, ldT,
				    T2, ldT,
			            T3, ldT,
			            Wmat, ldW );

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
