/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
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
#include <algorithm>
#include <cmath>

__device__ static double rocsolver_conj(double x)
{
    return (x);
}
__device__ static float rocsolver_conj(float x)
{
    return (x);
}
__device__ static rocblas_double_complex rocsolver_conj(rocblas_double_complex x)
{
    return (std::conj(x));
}
__device__ static rocblas_float_complex rocsolver_conj(rocblas_float_complex x)
{
    return (std::conj(x));
}

// ----------------
// solve X * L' = B
//
// L is N by N
// X is m by N
// B is m by N
//
// X is over-written by B
// ----------------
template <typename T, typename I, int N = 1>
__device__ static void
    trsm_Lower_Conj_Right_small(const I m, T const* const L, const I ldl, T* const B, const I ldb)
{
    auto idx2D = [](auto i, auto j, auto lda) { return (i + j * lda); };

    auto const X = B;
    auto const ldx = ldb;

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;
    auto const k_start = tid;
    auto const k_inc = nthreads;
    assert(hipBlockDim_z == 1);

    if(N == 1)
    {
        auto const el11 = *L;
        for(I k = k_start; k < m; k += k_inc)
        {
            B[k] = B[k] / rocsolver_conj(el11);
        }

        return;
    }

    // -----------------------------
    // [X1 X2] [L11'  L21'] = [B1  B2]
    //         [      L22']
    //
    // (1) X1 L11' = B1
    // (2) X1 * L21' + X2 * L22' = B2
    //
    // (2a) B2 = B2 - X1 * L21'
    // (2b) X2 * L22' = B2
    // -----------------------------

    auto constexpr n1 = N / 2;
    auto constexpr n2 = N - n1;

    T const* const L11 = L;
    T const* const L21 = L + idx2D(n1, 0, ldl);
    T const* const L22 = L + idx2D(n1, n1, ldl);

    T* const B1 = B;
    T* const B2 = B + idx2D(0, n1, ldb);
    T* const X1 = B1;
    T* const X2 = B2;

    // ----------------
    // (1) X1 L11' = B1
    // ----------------
    __syncthreads();

    {
        auto const ld1 = ldl;
        auto const ld2 = ldb;
        trsm_Lower_Conj_Right_small<T, I, n1>(m, L11, ld1, B1, ld2);
    }

    // -----------------------
    // (2a) B2 = B2 - X1 * L21'
    //
    // L21 is n2 by n1
    // L21' is n1 by n2
    //
    // B2 = m by n2
    // X1 = m by n1
    // -----------------------
    __syncthreads();

    for(auto j = j_start; j < n2; j += j_inc)
    {
        for(auto i = i_start; i < m; i += i_inc)
        {
            T csum = 0;

#pragma unroll
            for(auto k = 0; k < n1; k++)
            {
                auto const jk = j + k * ldl;
                auto const ik = i + k * ldx;
                csum += X1[ik] * rocsolver_conj(L21[jk]);
            }
            auto const ij = i + j * ldb;
            B2[ij] = B2[ij] - csum;
        }
    }

    __syncthreads();

    // -------------------
    // (2b) X2 * L22' = B2
    //
    // L22 is n2 by n2
    // B2 is m by n2
    // X2 is m by n2
    // -------------------
    {
        auto const ld1 = ldl;
        auto const ld2 = ldb;
        trsm_Lower_Conj_Right_small<T, I, n2>(m, L22, ld1, B2, ld2);
    }
}

// ----------------
// solve U' * X = B
// ----------------
template <typename T, typename I, int N = 1>
__device__ static void
    trsm_Upper_Conj_Left_small(const I nrhs, T const* const U, const I ldu, T* const B, const I ldb)
{
    auto idx2D = [](auto i, auto j, auto lda) { return (i + j * lda); };

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;
    auto const k_start = tid;
    auto const k_inc = nthreads;
    assert(hipBlockDim_z == 1);

    if(N == 1)
    {
        // -----------------------------
        // special case of 1 by 1 matrix
        // -----------------------------
        auto const u11 = *U;

        for(I k = k_start; k < nrhs; k += k_inc)
        {
            B[k] = B[k] / rocsolver_conj(u11);
        }
        return;
    }

    // -------------------------------
    // [ U11'      ] * [ X1 ] = [ B1 ]
    // [ U12'  U22']   [ X2 ]   [ B2 ]
    //
    // (1) U11' X1 = B1
    // (2) B2 <- B2 - U12' * X1
    // (3) U22' X2 = B2
    //
    // U11 is n1 by n1,
    // U12 is n1 by n2,  or U12' is n2 by n1
    // U22 is n2 by n2
    // -------------------------------
    auto constexpr n1 = N / 2;
    auto constexpr n2 = N - n1;
    auto const ldx = ldb;

    T* const B1 = B;
    T* const B2 = B + idx2D(n1, 0, ldb);
    T* const X1 = B1;
    T* const X2 = B2;

    T const* const U11 = U;
    T const* const U12 = U + idx2D(0, n1, ldu);
    T const* const U22 = U + idx2D(n1, n2, ldu);

    __syncthreads();

    // ----------------
    // (1) U11' X1 = B1
    // ----------------
    trsm_Upper_Conj_Left_small<T, I, n1>(nrhs, U11, ldu, B1, ldb);

    __syncthreads();

    //  --------------------
    // (2) B2 <- B2 - U12' * X1
    //
    //  note: U12' is n2 by n1
    //        B2 is n2 by nrhs
    //        X1 is n1 by nrhs
    //  --------------------
    {
        for(auto j = j_start; j < nrhs; j += j_inc)
        {
            for(auto i = i_start; i < n2; i += i_inc)
            {
                T csum = 0;

#pragma unroll
                for(I k = 0; k < n1; k++)
                {
                    auto const ki = k + i * ldu;
                    auto const kj = k + j * ldx;
                    csum += rocsolver_conj(U12[ki]) * X1[kj];
                }
                auto const ij = i + j * ldb;
                B2[ij] = B2[ij] - csum;
            }
        }
    }
    __syncthreads();

    // ----------------
    // (3) U22' X2 = B2
    // ----------------

    trsm_Upper_Conj_Left_small<T, I, n2>(nrhs, U22, ldu, B2, ldb);

    __syncthreads();
}

template <typename T, typename Treal, typename I>
__device__ static void herk_small(bool const is_upper,
                                  bool const is_trans,
                                  I const m,
                                  I const ksize,
                                  Treal const alpha,
                                  T const* const A,
                                  I const lda,
                                  Treal const beta,
                                  T* const C,
                                  I const ldc)
{
    // ----------------------------------------------------
    // C <- alpha * A * A' + beta * C,  if is_trans = false
    // C <- alpha * A' * A + beta * C,  if is_trans = true
    //
    // C is m by m,
    // A is m by k   if is_trans = false
    // A is k by m   if is_trans = true
    // ----------------------------------------------------

    bool const is_lower = (!is_upper);

    // -----------------------------------------
    // assume 2D grid of threads in thread block
    // -----------------------------------------

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;
    assert(hipBlockDim_z == 1);

    T const zero = static_cast<T>(0);
    T const one = static_cast<T>(1);
    bool const is_beta_zero = (beta == zero);
    bool const is_beta_one = (beta == one);

    __syncthreads();

    if(!is_beta_one)
    {
        for(auto j = j_start; j < m; j += j_inc)
        {
            for(auto i = i_start; i < m; i += i_inc)
            {
                bool const lower_part = (i >= j);
                bool const upper_part = (i <= j);
                bool const do_work = (is_upper && upper_part) || (is_lower && lower_part);
                if(do_work)
                {
                    auto const ij = i + j * (ldc);
                    bool const is_diag = (i == j);

                    C[ij] = is_beta_zero ? zero
                        : (is_diag)      ? beta * std::real(C[ij])
                                         : beta * C[ij];
                }
            }
        }
    }

    __syncthreads();

    for(auto j = j_start; j < m; j += j_inc)
    {
        for(auto i = i_start; i < m; i += i_inc)
        {
            bool const lower_part = (i >= j);
            bool const upper_part = (i <= j);
            bool const do_work = (is_upper && upper_part) || (is_lower && lower_part);
            if(do_work)
            {
                T csum = zero;
                for(I k = 0; k < ksize; k++)
                {
                    // ------------------------------------------------------
                    // C += alpha * A' * A,  if (is_trans)
                    // C(i,j) += alpha * sum( conj(A(k,i)) * A(k,j), over k)
                    // ------------------------------------------------------

                    // ------------------------------------------------------
                    // C += alpha * A * A', if (!is_trans)
                    // C(i,j) += alpha * sum( A(i,k) * conj(A(j,k)), over k )
                    // ------------------------------------------------------
                    auto const ki = k + (i * lda);
                    auto const kj = k + (j * lda);
                    auto const ik = i + (k * lda);
                    auto const jk = j + (k * lda);
                    csum += (is_trans) ? (rocsolver_conj(A[ki]) * A[kj])
                                       : (A[ik] * rocsolver_conj(A[jk]));
                }

                auto const ij = i + (j * ldc);
                C[ij] += alpha * csum;
            }
        }
    }
    __syncthreads();
}

/**
 * ------------------------------------------------
 * perform Cholesky factorization for N by N matrix.
 * using a conceptually recursive block formulation
 * note: this function executes within a single thread block.
 * ------------------------------------------------
**/

template <typename T, typename Treal, typename I, int N = 1>
__device__ static void potf2_small(bool is_upper, T* A, const I lda, I* info, I ioffset)
{
    auto idx2D = [](auto i, auto j, auto lda) { return (i + j * lda); };

    // ---------------------------------
    // assume A is already loaded in LDS
    // works on a single thread block
    // ---------------------------------
    assert(info != nullptr);
    assert(A != nullptr);

    auto constexpr n1 = N / 2;
    auto constexpr n2 = N - n1;

    T* const A11 = A;
    T* const A21 = A + idx2D(n1, 0, lda);
    T* const A12 = A + idx2D(0, n1, lda);
    T* const A22 = A + idx2D(n1, n1, lda);

    if(N == 1)
    {
        T const aii = *A;
        bool const isok = (std::real(aii) >= 0); // note test ignores imaginary part
        *info = (isok) ? 0 : 1 + ioffset;
        if(isok)
        {
            *A = std::sqrt(std::abs(aii));
        }
        return;
    }

    // ----------
    // factor A11
    // ----------
    {
        potf2_small<T, Treal, I, n1>(is_upper, A11, lda, info, ioffset);
    }

    if(is_upper)
    {
        // --------------------------------------
        // U' * U = A
        //
        // [U11'      ] * [U11  U12] = [A11  A12]
        // [U12'  U22']   [     U22]   [A12' A22]
        //
        // (1) U11' * U11 = A11
        // (2) U11' * U12 = A12
        // (3) U22' * U22 = (A22 - U12' * U12)
        // --------------------------------------

        T* const U11 = A11;
        T* const U12 = A12;
        auto const ldu = lda;

        {
            // ----------------------
            // solve U11' * U12 = A12
            //
            // U11 is n1 by n1
            // U12 is n1 by n2
            // ----------------------
            auto const ld1 = ldu;
            auto const ld2 = ldu;
            auto const nrhs = n2;
            trsm_Upper_Conj_Left_small<T, I, n1>(nrhs, U11, ld1, U12, ld2);
        }

        {
            // -------------------------------
            // U22' * U22 = (A22 - U12' * U12)
            //
            // A22 is n2 by n2
            // U12 is n1 by n2, so U12' is n2 by n1
            //
            // -------------------------------
            auto constexpr mm = n2;
            auto constexpr kk = n1;
            bool constexpr is_trans = true;
            Treal const alpha = -1;
            Treal const beta = 1;
            auto const ld1 = ldu;
            auto const ld2 = lda;

            herk_small<T, Treal, I>(is_upper, is_trans, mm, kk, alpha, U12, ld1, beta, A22, ld2);
        }
    }
    else
    {
        // --------------------------------------
        // L * L' = A
        //
        //
        // [L11    ] * [L11'  L21'] = [A11  A21']
        // [L21 L22]   [      L22']   [A21  A22 ]
        //
        // (1) L11 * L11' = A11
        // (2) L21 * L11' = A21
        // (3) L22 * L22' = (A22 - L21 * L21')
        // --------------------------------------

        T* const L11 = A11;
        T* const L21 = A21;
        auto const ldl = lda;

        {
            //  ----------------
            //  L21 * L11' = A21
            //
            //  L21 is n2 by n1
            //  L11 is n1 by n1
            //  ----------------
            auto const ld1 = ldl;
            auto const ld2 = lda;
            trsm_Lower_Conj_Right_small<T, I, n1>(n2, L11, ld1, A21, ld2);
        }

        {
            //  -------------------------------
            //  L22 * L22' = (A22 - L21 * L21')
            //  -------------------------------
            auto constexpr mm = n2;
            auto constexpr kk = n1;
            bool constexpr is_trans = false;
            Treal const alpha = -1;
            Treal const beta = 1;

            auto const ld1 = lda;
            auto const ld2 = lda;

            herk_small<T, Treal, I>(is_upper, is_trans, mm, kk, alpha, L21, ld1, beta, A22, ld2);
        }
    }

    {
        auto const linfo = info + n1;
        auto const loffset = ioffset + n1;
        potf2_small<T, Treal, I, n2>(is_upper, A22, lda, linfo, loffset);
    }
}

template <typename T, typename I>
__device__ static void
    potf2_simple(bool const is_upper, I const n, T* const A, I const lda, I* const info)
{
    bool const is_lower = (!is_upper);

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;
    assert(hipBlockDim_z == 1);

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto const j0_start = tid;
    auto const j0_inc = nthreads;

    auto idx2D = [](auto i, auto j, auto lda) { return (i + j * lda); };

    if(is_lower)
    {
        // ---------------------------------------------------
        // [  l11     ]  * [ l11'   vl21' ]  =  [ a11       ]
        // [ vl21  L22]    [        L22' ]     [ va21, A22 ]
        //
        //
        //   assume l11 is scalar 1x1 matrix
        //
        //   (1) l11 * l11' = a11 =>  l11 = sqrt( abs(a11) ), scalar computation
        //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
        //   (3) L22 * L22' + vl21 * vl21' = A22
        //
        //   (3a) A22 = A22 - vl21 * vl21',  symmetric rank-1 update
        //   (3b) L22 * L22' = A22,   cholesky factorization, tail recursion
        // ---------------------------------------------------

        for(I kcol = 0; kcol < n; kcol++)
        {
            auto kk = idx2D(kcol, kcol, lda);
            auto const akk = std::real(A[kk]);
            bool const isok = (akk > 0) && (std::isfinite(akk));
            if(!isok)
            {
                if(tid == 0)
                {
                    A[kk] = akk;
                    // Fortran 1-based index
                    *info = (*info == 0) ? kcol + 1 : (*info);
                }
                break;
            }

            auto const lkk = std::sqrt(akk);
            if(tid == 0)
            {
                A[kk] = lkk;
            }

            __syncthreads();

            // ------------------------------------------------------------
            //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
            // ------------------------------------------------------------

            auto const inv_lkk = 1 / rocsolver_conj(lkk);
            for(I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc)
            {
                auto const j0k = idx2D(j0, kcol, lda);
                A[j0k] *= inv_lkk;
            }

            __syncthreads();

            // ------------------------------------------------------------
            //   (3a) A22 = A22 - vl21 * vl21',  symmetric rank-1 update
            //
            //   note: update lower triangular part
            // ------------------------------------------------------------

            for(I j = (kcol + 1) + j_start; j < n; j += j_inc)
            {
                auto const vj = A[idx2D(j, kcol, lda)];
                for(I i = (kcol + 1) + i_start; i < n; i += i_inc)
                {
                    bool const lower_part = (i >= j);
                    if(lower_part)
                    {
                        auto const vi = A[idx2D(i, kcol, lda)];
                        auto const ij = idx2D(i, j, lda);
                        A[ij] = A[ij] - vi * rocsolver_conj(vj);
                    };
                }
            }

            __syncthreads();

        } // end for kcol
    }
    else
    {
        // --------------------------------------------------
        // [u11'        ] * [u11    vU12 ] = [ a11     vA12 ]
        // [vU12'   U22']   [       U22  ]   [ vA12'   A22  ]
        //
        // (1) u11' * u11 = a11 =?  u11 = sqrt( abs( a11 ) )
        // (2) vU12' * u11 = vA12', or u11' * vU12 = vA12
        //     or vU12 = vA12/u11'
        // (3) vU12' * vU12 + U22'*U22 = A22
        //
        // (3a) A22 = A22 - vU12' * vU12
        // (3b) U22' * U22 = A22,  cholesky factorization, tail recursion
        // --------------------------------------------------

        for(I kcol = 0; kcol < n; kcol++)
        {
            auto const kk = idx2D(kcol, kcol, lda);
            auto const akk = std::real(A[kk]);
            bool const isok = (akk > 0) && (std::isfinite(akk));
            if(!isok)
            {
                A[kk] = akk;
                if(tid == 0)
                {
                    // Fortran 1-based index
                    *info = (*info == 0) ? kcol + 1 : (*info);
                }

                break;
            }

            auto const ukk = std::sqrt(akk);
            if(tid == 0)
            {
                A[kk] = ukk;
            }
            __syncthreads();

            // ----------------------------------------------
            // (2) vU12' * u11 = vA12', or u11' * vU12 = vA12
            // ----------------------------------------------
            auto const inv_ukk = 1 / rocsolver_conj(ukk);
            for(I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc)
            {
                auto const kj0 = idx2D(kcol, j0, lda);
                A[kj0] *= inv_ukk;
            }

            __syncthreads();

            // -----------------------------
            // (3a) A22 = A22 - vU12' * vU12
            //
            // note: update upper triangular part
            // -----------------------------
            for(I j = (kcol + 1) + j_start; j < n; j += j_inc)
            {
                auto const vj = A[idx2D(kcol, j, lda)];
                for(I i = (kcol + 1) + i_start; i < n; i += i_inc)
                {
                    bool const upper_part = (i <= j);
                    if(upper_part)
                    {
                        auto const vi = A[idx2D(kcol, i, lda)];
                        auto const ij = idx2D(i, j, lda);

                        A[ij] = A[ij] - rocsolver_conj(vi) * vj;
                    };
                }
            }

            __syncthreads();

        } // end for kcol
    }

    __syncthreads();
}

template <typename T, typename U>
ROCSOLVER_KERNEL void potf2_lds(const bool is_upper,
                                const rocblas_int n,
                                U AA,
                                const rocblas_int shiftA,
                                const rocblas_stride strideA,
                                const rocblas_int lda,
                                rocblas_int* const info)
{
    bool const is_lower = (!is_upper);

    using Treal = decltype(std::real(T{}));

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;
    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;
    assert(hipBlockDim_z == 1);

    // --------------------------------
    // note hipGridDim_z == batch_count
    // --------------------------------
    auto const bid = hipBlockIdx_z;
    assert(AA != nullptr);

    T* const A = (AA != nullptr) ? load_ptr_batch(AA, bid, shiftA, strideA) : nullptr;

    assert(info != nullptr);
    rocblas_int* const info_bid = (info == nullptr) ? nullptr : &(info[bid]);

    assert(A != nullptr);

    auto idx2D = [](auto i, auto j, auto lda) { return (i + j * static_cast<int64_t>(lda)); };

    // -----------------------------------------
    // assume n by n matrix will fit in LDS cache
    // -----------------------------------------

    auto const ld_Ash = n;
    size_t constexpr LDS_MAXIMUM_SIZE = 64 * 1024;

    bool const use_lds = (sizeof(T) * ld_Ash * n <= LDS_MAXIMUM_SIZE);
    __shared__ T Ash[LDS_MAXIMUM_SIZE / sizeof(T)];

    T const zero = static_cast<T>(0);
    // ------------------------------------
    // copy n by n matrix into shared memory
    // ------------------------------------
    __syncthreads();

    if(use_lds)
    {
        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = i_start; i < n; i += i_inc)
            {
                bool const lower_part = (i >= j);
                bool const upper_part = (i <= j);
                bool const do_assignment = (is_upper && upper_part) || (is_lower && lower_part);

                Ash[i + j * ld_Ash] = (do_assignment) ? A[idx2D(i, j, lda)] : zero;
            }
        }
    }
    __syncthreads();

    {
        T* const Amat = (use_lds) ? Ash : A;
        auto const ldAmat = (use_lds) ? ld_Ash : lda;
        potf2_simple<T, rocblas_int>(is_upper, n, Amat, ldAmat, info_bid);
    }

    __syncthreads();

    // -------------------------------------
    // copy n by n matrix into global memory
    // -------------------------------------
    if(use_lds)
    {
        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = i_start; i < n; i += i_inc)
            {
                bool const lower_part = (i >= j);
                bool const upper_part = (i <= j);
                bool const do_assignment = (is_upper && upper_part) || (is_lower && lower_part);
                if(do_assignment)
                {
                    auto const ij = idx2D(i, j, lda);
                    A[ij] = Ash[i + j * ld_Ash];
                }
            }
        }
    }
    __syncthreads();
}
