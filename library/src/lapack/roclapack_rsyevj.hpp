/************************************************************************
 * Derived from
 * Gotlub & Van Loan (1996). Matrix Computations (3rd ed.).
 *     John Hopkins University Press.
 *     Section 8.4.
 * and
 * Hari & Kovac (2019). On the Convergence of Complex Jacobi Methods.
 *     Linear and Multilinear Algebra 69(3), p. 489-514.
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc.
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
 * ***********************************************************************/

#pragma once

#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "roclapack_syev_heev.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

/************** Kernels and device functions for small size*******************/
/*****************************************************************************/

#define RSYEVJ_BDIM 1024 // Max number of threads per thread-block used in syevj_small kernel

// --------------------------------------------------
// need to fit n by n copy of A,  cosines, sines, and
// optional n by n copy of V
// --------------------------------------------------
#define RSYEVJ_BLOCKED_SWITCH(T, need_V)            \
    (((need_V) && (sizeof(T) == 4))           ? 90  \
         : ((need_V) && (sizeof(T) == 8))     ? 62  \
         : ((need_V) && (sizeof(T) == 16))    ? 44  \
         : ((!(need_V)) && (sizeof(T) == 4))  ? 126 \
         : ((!(need_V)) && (sizeof(T) == 8))  ? 88  \
         : ((!(need_V)) && (sizeof(T) == 16)) ? 62  \
                                              : 32)

// ------------------------------------------------------------------------------------------
// symmetrize_matrix make the square matrix a symmetric or hermitian matrix
// if (uplo == 'U') use the entries in upper triangular part to set the lower triangular part
// if (uplo == 'L') use the entries in lower triangular part to set the upper triangular part
// ------------------------------------------------------------------------------------------
template <typename T, typename I>
static __device__ void symmetrize_matrix_body(char const uplo,
                                              I const n,
                                              T* A_,
                                              I const lda,
                                              I const i_start,
                                              I const i_inc,
                                              I const j_start,
                                              I const j_inc)
{
    bool const use_upper = (uplo == 'U') || (uplo == 'u');
    bool const use_lower = (uplo == 'L') || (uplo == 'l');

    auto A = [=](auto i, auto j) -> T& { return (A_[idx2D(i, j, lda)]); };

    for(auto j = j_start; j < n; j += j_inc)
    {
        for(auto i = i_start; i < n; i += i_inc)
        {
            bool const is_strictly_lower = (i > j);
            bool const is_strictly_upper = (i < j);
            bool const is_diagonal = (i == j);

            if(use_upper)
            {
                if(is_strictly_upper)
                {
                    A(j, i) = conj(A(i, j));
                }
            }

            if(use_lower)
            {
                if(is_strictly_lower)
                {
                    A(j, i) = conj(A(i, j));
                }
            }

            if(is_diagonal)
            {
                A(i, i) = (A(i, i) + conj(A(i, i))) / 2;
            }
        }
    }
}

// ----------------------------------------
// copy submatrix from matrix A to matrix B
// if (uplo == 'U') copy only the upper triangular part
// if (uplo == 'L') copy only the lower triangular part
// otherwise, copy the entire m by n  matrix
// ----------------------------------------
template <typename T, typename I>
static __device__ void lacpy_body(char const uplo,
                                  I const m,
                                  I const n,
                                  T const* const A_,
                                  I const lda,
                                  T* B_,
                                  I const ldb,
                                  I const i_start,
                                  I const i_inc,
                                  I const j_start,
                                  I const j_inc)
{
    bool const use_upper = (uplo == 'U') || (uplo == 'u');
    bool const use_lower = (uplo == 'L') || (uplo == 'l');
    bool const use_full = (!use_upper) && (!use_lower);

    auto A = [=](auto i, auto j) -> const T& { return (A_[idx2D(i, j, lda)]); };

    auto B = [=](auto i, auto j) -> T& { return (B_[idx2D(i, j, ldb)]); };

    for(auto j = j_start; j < n; j += j_inc)
    {
        for(auto i = i_start; i < m; i += i_inc)
        {
            bool const is_diagonal = (i == j);
            bool const is_strictly_upper = (i < j);
            bool const is_strictly_lower = (i > j);

            bool const is_upper = (is_diagonal || is_strictly_upper);
            bool const is_lower = (is_diagonal || is_strictly_lower);

            bool const do_assignment = use_full || (use_upper && is_upper) || (use_lower && is_lower);
            if(do_assignment)
            {
                B(i, j) = A(i, j);
            }
        }
    }
}

// --------------------------------------------------------------------
// laset set off-diagonal entries to alpha,
// and diagonal entries to beta
// if (uplo == 'U') set only the upper triangular part
// if (uplo == 'L') set only the lower triangular part
// otherwise, set the whole m by n matrix
// --------------------------------------------------------------------
template <typename T, typename I>
static __device__ void laset_body(char const uplo,
                                  I const m,
                                  I const n,
                                  T const alpha,
                                  T const beta,
                                  T* A_,
                                  I const lda,
                                  I const i_start,
                                  I const i_inc,
                                  I const j_start,
                                  I const j_inc)
{
    // set offdiagonal to alpha
    // set diagonal to beta

    bool const use_upper = (uplo == 'U') || (uplo == 'u');
    bool const use_lower = (uplo == 'L') || (uplo == 'l');
    bool const use_full = (!use_upper) && (!use_lower);

    auto A = [=](auto i, auto j) -> T& { return (A_[idx2D(i, j, lda)]); };

    for(I j = j_start; j < n; j += j_inc)
    {
        for(I i = i_start; i < m; i += i_inc)
        {
            bool const is_diag = (i == j);
            bool const is_strictly_upper = (i < j);
            bool const is_strictly_lower = (i > j);

            bool const do_assignment
                = (use_full || (use_lower && is_strictly_lower) || (use_upper && is_strictly_upper));

            if(do_assignment)
            {
                A(i, j) = alpha;
            }

            if(is_diag)
            {
                A(i, i) = beta;
            }
        }
    }
}

// --------------------------------------
// index calculations for compact storage
// --------------------------------------
template <typename I>
static __device__ I idx_upper(I const i, I const j, I const n)
{
    return (i + (j * (j + 1)) / 2);
};

template <typename I>
static __device__ I idx_lower(I const i, I const j, I const n)
{
    return ((i - j) + (j * (2 * n + 1 - j)) / 2);
};

// -----------------------------------------------------------------------------
// dlaev2 computes the eigen decomposition of a 2x2 hermitian or symmetric matrix
// [ cs1  sn1 ]  [ a       b]  [ cs1   -sn1 ] = [ rt1    0  ]
// [-sn1  cs1 ]  [ b       c]  [ sn1    cs1 ]   [ 0      rt2]
// -----------------------------------------------------------------------------
template <typename S>
static __device__ void dlaev2(S const a, S const b, S const c, S& rt1, S& rt2, S& cs1, S& sn1)
{
    double const one = 1.0;
    double const two = 2.0;
    double const zero = 0;
    double const half = 0.5;

    auto abs = [](auto x) { return ((x >= 0) ? x : (-x)); };
    auto sqrt = [](auto x) { return (std::sqrt(x)); };
    auto square = [](auto x) { return (x * x); };
    auto dble = [](auto x) { return (static_cast<double>(x)); };

    int sgn1, sgn2;
    S ab, acmn, acmx, acs, adf, cs, ct, df, rt, sm, tb, tn;

    sm = a + c;
    df = a - c;
    adf = abs(df);
    tb = b + b;
    ab = abs(tb);
    if(abs(a) > abs(c))
    {
        acmx = a;
        acmn = c;
    }
    else
    {
        acmx = c;
        acmn = a;
    }
    if(adf > ab)
    {
        rt = adf * sqrt(one + square(ab / adf));
    }
    else if(adf < ab)
    {
        rt = ab * sqrt(one + square(adf / ab));
    }
    else
    {
        //
        //        includes case ab=adf=0
        //
        rt = ab * sqrt(two);
    }
    if(sm < zero)
    {
        rt1 = half * (sm - rt);
        sgn1 = -1;
        //
        //        order of execution important.
        //        to get fully accurate smaller eigenvalue,
        //        next line needs to be executed in higher precision.
        //
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
    }
    else if(sm > zero)
    {
        rt1 = half * (sm + rt);
        sgn1 = 1;
        //
        //        order of execution important.
        //        to get fully accurate smaller eigenvalue,
        //        next line needs to be executed in higher precision.
        //
        rt2 = (dble(acmx) / dble(rt1)) * dble(acmn) - (dble(b) / dble(rt1)) * dble(b);
    }
    else
    {
        //
        //        includes case rt1 = rt2 = 0
        //
        rt1 = half * rt;
        rt2 = -half * rt;
        sgn1 = 1;
    }
    //
    //     compute the eigenvector
    //
    if(df >= zero)
    {
        cs = df + rt;
        sgn2 = 1;
    }
    else
    {
        cs = df - rt;
        sgn2 = -1;
    }
    acs = abs(cs);
    if(acs > ab)
    {
        ct = -tb / cs;
        sn1 = one / sqrt(one + ct * ct);
        cs1 = ct * sn1;
    }
    else
    {
        if(ab == zero)
        {
            cs1 = one;
            sn1 = zero;
        }
        else
        {
            tn = -cs / tb;
            cs1 = one / sqrt(one + tn * tn);
            sn1 = tn * cs1;
        }
    }
    if(sgn1 == sgn2)
    {
        tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    return;
}

// -----------------------------------------------------------------------------
// zlaev2 computes the eigen decomposition of a 2x2 hermitian or symmetric matrix
// [cs1  conj(sn1) ]  [ a        b]  [ cs1   -conj(sn1) ] = [ rt1    0  ]
// [-sn1  cs1      ]  [ conj(b) c ]  [ sn1    cs1       ]   [ 0      rt2]
// -----------------------------------------------------------------------------
template <typename T, typename S>
__device__ static void zlaev2(T const a, T const b, T const c, S& rt1, S& rt2, S& cs1, T& sn1)
{
    S const zero = 0.0;
    S const one = 1.0;

    S t;
    T w;

    auto abs = [](auto x) { return (std::abs(x)); };
    auto dble = [](auto x) { return (static_cast<S>(std::real(x))); };
    auto dconjg = [](auto x) { return (conj(x)); };

    if(abs(b) == zero)
    {
        w = one;
    }
    else
    {
        w = dconjg(b) / abs(b);
    }
    dlaev2(dble(a), abs(b), dble(c), rt1, rt2, cs1, t);
    sn1 = w * t;
    return;
}

template <typename T, typename S>
__device__ static void laev2(T const a, T const b, T const c, S& rt1, S& rt2, S& cs1, T& sn1)
{
    bool const is_complex = rocblas_is_complex<T>;
    if constexpr(is_complex)
    {
        zlaev2(a, b, c, rt1, rt2, cs1, sn1);
    }
    else
    {
        dlaev2(a, b, c, rt1, rt2, cs1, sn1);
    }
}

#if(0)
template <>
__device__ static void laev2(double const a,
                             double const b,
                             double const c,
                             double& rt1,
                             double& rt2,
                             double& cs1,
                             double& sn1)
{
    dlaev2(a, b, c, rt1, rt2, cs1, sn1);
}

template <>
__device__ static void
    laev2(float const a, float const b, float const c, float& rt1, float& rt2, float& cs1, float& sn1)
{
    dlaev2(a, b, c, rt1, rt2, cs1, sn1);
}
#endif

// -------------------------------------
// calculate the 2-norm of n by n matrix
// dwork(:) is array of length n
// -------------------------------------
template <typename T, typename I, typename S>
static __device__ double cal_norm(I const n,
                                  T const* const A_,
                                  I const lda,
                                  S* const dwork,
                                  I const i_start,
                                  I const i_inc,
                                  I const j_start,
                                  I const j_inc,
                                  bool const include_diagonal)
{
    auto A = [=](auto i, auto j) -> const T& { return (A_[i + j * lda]); };

    // ------------------
    // initialize dwork(:)
    // ------------------
    if(j_start == 0)
    {
        for(auto i = i_start; i < n; i += i_inc)
        {
            dwork[i] = 0;
        };
    }
    __syncthreads();

    for(auto j = j_start; j < n; j += j_inc)
    {
        for(auto i = i_start; i < n; i += i_inc)
        {
            bool const is_diag = (i == j);
            bool const need_accumulate = (!is_diag) || (is_diag && include_diagonal);
            if(need_accumulate)
            {
                auto const aij = A(i, j);
                auto const aij_norm = std::norm(aij);
                atomicAdd(&(dwork[i]), aij_norm);
            }
        }
    }
    __syncthreads();

    bool const is_root = (i_start == 0) && (j_start == 0);
    if(is_root)
    {
        double dnorm = 0;
        for(auto i = 0; i < n; i++)
        {
            dnorm += dwork[i];
        };

        dwork[0] = std::sqrt(dnorm);
    }

    __syncthreads();

    return (dwork[0]);
}

/** RSYEVJ_SMALL_KERNEL/RUN_RSYEVJ applies the Jacobi eigenvalue algorithm to matrices of size
    n <= RSYEVJ_BLOCKED_SWITCH(T,need_V). For each off-diagonal element A(p,q), a Jacobi rotation J is
    calculated so that (J'AJ)(p,q) = 0. J only affects rows i and j, and J' only affects
    columns (and rows) p and q. Therefore, (n / 2) rotations can be computed and applied
    in parallel, so long as the rotations do not conflict. We use a precompute tournament schedule
    for n players to obtain the set of independent pairs (p,q) that do not conflict.

    (Call the rsyevj_small_kernel with batch_count groups in z, of dim = ddx * ddy threads in x.
	Then, the run_syevj device function will be run by all threads organized in a ddx-by-ddy array.
	Normally, ddx = 32, and ddy <= min(32,ceil(n / 2)). Any index pair (p,q)  with invalid values
	for p or q will be skipped.

 **/

template <typename T, typename I, typename S>
__device__ void run_rsyevj(const I dimx,
                           const I dimy,
                           const I tix,
                           const I tiy,
                           const rocblas_esort esort,
                           const rocblas_evect evect,
                           const rocblas_fill uplo,
                           const I n,
                           T* dA_,
                           const I lda,
                           const S abstol,
                           const S eps,
                           S* residual,
                           const I max_sweeps,
                           I* n_sweeps,
                           S* W,
                           I* info,
                           T* A_,
                           T* V_,
                           S* cosines_res,
                           T* sines_diag,
                           I const* const schedule_)

{
    // ---------------------------------------------
    // ** NOTE **  n can be an odd number
    // but the tournament schedule is generated for
    // n_even players
    // ---------------------------------------------
    auto const n_even = n + (n % 2);

    auto const cosine = cosines_res;
    auto const sine = sines_diag;
    const bool need_sort = (esort != rocblas_esort_none);
    bool const is_upper = (uplo == rocblas_fill_upper);
    bool const need_V = (evect != rocblas_evect_none) && (V_ != nullptr);

    auto const ldv = n;

    auto const i_start = tix;
    auto const i_inc = dimx;
    auto const j_start = tiy;
    auto const j_inc = dimy;

    // ---------------------------------------
    // reuse storage
    // ** NOTE ** need to  use both arrays of cosine and sine
    // ---------------------------------------
    S* const dwork = cosine;

    // ----------------------------------------------------
    // schedule_(:)  is array of size n_even * (n_even - 1)
    // that contains the tournament schedule for n_even number
    // of players
    // ----------------------------------------------------
    auto schedule = [=](auto i1, auto itable, auto iround) {
        return (schedule_[i1 + itable * 2 + iround * n_even]);
    };

    auto V = [=](auto i, auto j) -> T& { return (V_[i + j * ldv]); };
    auto A = [=](auto i, auto j) -> T& { return (A_[i + j * lda]); };

    auto const num_rounds = (n_even - 1);
    auto const ntables = (n_even / 2);

    {
        bool const is_same = (dA_ == A_);
        if(!is_same)
        {
            // -------------------------------------------------
            // copy dA from device memory to A_ in shared memory
            // -------------------------------------------------
            char const c_uplo = (is_upper) ? 'U' : 'L';
            I const mm = n;
            I const nn = n;
            I const ld1 = lda;
            I const ld2 = mm;
            lacpy_body(c_uplo, mm, nn, dA_, ld1, A_, ld2, i_start, i_inc, j_start, j_inc);

            __syncthreads();

            // ----------------
            // symmetrize matrix
            // ----------------

            symmetrize_matrix_body(c_uplo, nn, A_, ld2, i_start, i_inc, j_start, j_inc);

            __syncthreads();
        }
    }

    // -------------------------------------------
    // set V to identity if computing eigenvectors
    // -------------------------------------------
    if(need_V)
    {
        char const c_uplo = 'A';
        T const alpha_offdiag = 0;
        T const beta_diag = 1;
        auto const mm = n;
        auto const nn = n;
        auto const ld1 = mm;

        laset_body(c_uplo, mm, nn, alpha_offdiag, beta_diag, A_, ld1, i_start, i_inc, j_start, j_inc);

        __syncthreads();
    }

    double norm_A = 0;
    {
        bool const need_diagonal = true;
        I ld1 = n;
        norm_A = cal_norm(n, A_, ld1, dwork, i_start, i_inc, j_start, j_inc, need_diagonal);
    }

    bool has_converged = false;

    // ----------------------------------------------------------
    // NOTE: need to preserve value of isweep outside of for loop
    // ----------------------------------------------------------
    I isweep = 0;

    for(isweep = 0; isweep < max_sweeps; isweep++)
    {
        // ----------------------------------------------------------
        // check convergence by computing norm of off-diagonal matrix
        // ----------------------------------------------------------
        __syncthreads();

        bool const need_diagonal = false;
        auto const norm_offdiag
            = cal_norm(n, A_, lda, dwork, i_start, i_inc, j_start, j_inc, need_diagonal);

        has_converged = (norm_offdiag <= abstol * norm_A);
        __syncthreads();

        if(has_converged)
        {
            break;
        };

        for(auto iround = 0; iround < num_rounds; iround++)
        {
            __syncthreads();
            for(auto j = j_start; j < ntables; j += j_inc)
            {
                // --------------------------
                // get independent pair (p,q)
                // --------------------------

                auto const p = schedule(0, j, iround);
                auto const q = schedule(1, j, iround);

                // -------------------------------------
                // if n is an odd number, then just skip
                // operation for invalid values
                // -------------------------------------
                bool const is_valid_pq = (0 <= q) && (q < n) && (0 <= p) && (p < n);

                if(!is_valid_pq)
                {
                    continue;
                };

                auto const App = A(p, p);
                auto const Apq = A(p, q);
                auto const Aqq = A(q, q);

                // ----------------------------------------------------------------------
                // [ cs1  conj(sn1) ][ App        Apq ] [cs1   -conj(sn1)] = [rt1   0   ]
                // [-sn1  cs1       ][ conj(Apq)  Aqq ] [sn1    cs1      ]   [0     rt2 ]
                // ----------------------------------------------------------------------
                T sn1 = 0;
                S cs1 = 0;
                T rt1 = 0;
                T rt2 = 0;
                laev2(App, Apq, Aqq, rt1, rt2, cs1, sn1);

                if(i_start == 0)
                {
                    cosine[j] = cs1;
                    sine[j] = sn1;
                }

                // ----------------------------
                // update rows p, q in matrix A
                // ----------------------------
                for(auto i = i_start; i < n; i += i_inc)
                {
                    auto const Api = A(p, i);
                    auto const Aqi = A(q, i);

                    A(p, i) = cs1 * Api + conj(sn1) * Aqi;
                    A(q, i) = -sn1 * Api + cs1 * Aqi;
                }

            } // end for j

            // ------------------------------------------------
            // need to wait for all row updates to be completed
            // ------------------------------------------------
            __syncthreads();

            for(auto j = j_start; j < ntables; j += j_inc)
            {
                // --------------------------
                // get independent pair (p,q)
                // --------------------------

                auto const p = schedule(0, j, iround);
                auto const q = schedule(1, j, iround);

                // -------------------------------------
                // if n is an odd number, then just skip
                // operation for invalid values
                // -------------------------------------
                bool const is_valid_pq = (0 <= q) && (q < n) && (0 <= p) && (p < n);
                if(!is_valid_pq)
                {
                    continue;
                };

                // ----------------------------------------------------------------------
                // [ cs1  conj(sn1) ][ App        Apq ] [cs1   -conj(sn1)] = [rt1   0   ]
                // [-sn1  cs1       ][ conj(Apq)  Aqq ] [sn1    cs1      ]   [0     rt2 ]
                // ----------------------------------------------------------------------

                auto const cs1 = cosine[j];
                auto const sn1 = sine[j];
                auto const sn2 = -conj(sn1);

                // -------------------------------
                // update columns p, q in matrix A
                // -------------------------------
                for(auto i = i_start; i < n; i += i_inc)
                {
                    {
                        auto const Aip = A(i, p);
                        auto const Aiq = A(i, q);

                        A(i, p) = Aip * cs1 + Aiq * sn1;
                        A(i, q) = Aip * sn2 + Aiq * cs1;
                    }

                    // update eigen vectors
                    if(need_V)
                    {
                        auto const Vip = V(i, p);
                        auto const Viq = V(i, q);
                        V(i, p) = Vip * cs1 + Viq * sn1;
                        V(i, q) = Vip * sn2 + Viq * cs1;
                    }
                }
            } // end for j
            __syncthreads();
        } // end for iround

    } // for isweeps

    // -----------------
    // copy eigen values
    // -----------------
    if(j_start == 0)
    {
        for(auto i = i_start; i < n; i += i_inc)
        {
            W[i] = A(i, i);
        }
    }
    __syncthreads();

    *info = (has_converged) ? 0 : 1;
    *n_sweeps = (has_converged) ? isweep : max_sweeps;

    // -----------------
    // sort eigen values
    // -----------------
    if(need_sort)
    {
        I* const map = (I*)dwork;
        shell_sort(n, W, map);
        __syncthreads();

        if(need_V)
        {
            permute_swap(n, V_, ldv, map);
            __syncthreads();
        }
    }

    return;
}

#if(0)
{
    // local variables
    S c, mag, f, g, r, s;
    T s1, s2, aij, temp1, temp2;
    rocblas_int i, j;
    rocblas_int sweeps = 0;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;
    S local_res = 0;
    S local_diag = 0;

    if(tiy == 0)
    {
        // copy A to Acpy, set A to identity (if calculating eigenvectors), and calculate off-diagonal
        // squared Frobenius norm (first by column/row then sum)
        if(uplo == rocblas_fill_upper)
        {
            for(i = tix; i < n; i += dimx)
            {
                aij = A[i + i * lda];
                local_diag += std::norm(aij);
                Acpy[i + i * n] = aij;

                if(evect != rocblas_evect_none)
                    A[i + i * lda] = 1;

                for(j = n - 1; j > i; j--)
                {
                    aij = A[i + j * lda];
                    local_res += 2 * std::norm(aij);
                    Acpy[i + j * n] = aij;
                    Acpy[j + i * n] = conj(aij);

                    if(evect != rocblas_evect_none)
                    {
                        A[i + j * lda] = 0;
                        A[j + i * lda] = 0;
                    }
                }
            }
        }
        else
        {
            for(i = tix; i < n; i += dimx)
            {
                aij = A[i + i * lda];
                local_diag += std::norm(aij);
                Acpy[i + i * n] = aij;

                if(evect != rocblas_evect_none)
                    A[i + i * lda] = 1;

                for(j = 0; j < i; j++)
                {
                    aij = A[i + j * lda];
                    local_res += 2 * std::norm(aij);
                    Acpy[i + j * n] = aij;
                    Acpy[j + i * n] = conj(aij);

                    if(evect != rocblas_evect_none)
                    {
                        A[i + j * lda] = 0;
                        A[j + i * lda] = 0;
                    }
                }
            }
        }
        cosines_res[tix] = local_res;
        sines_diag[tix] = local_diag;

        // initialize top/bottom pairs
        for(i = tix; i < half_n; i += dimx)
        {
            top[i] = i * 2;
            bottom[i] = i * 2 + 1;
        }
    }
    __syncthreads();

    // set tolerance
    local_res = 0;
    local_diag = 0;
    for(i = 0; i < dimx; i++)
    {
        local_res += cosines_res[i];
        local_diag += std::real(sines_diag[i]);
    }
    S tolerance = (local_res + local_diag) * abstol * abstol;
    S small_num = get_safemin<S>() / eps;

    // execute sweeps
    rocblas_int count = (half_n - 1) / dimx + 1;
    while(sweeps < max_sweeps && local_res > tolerance)
    {
        // for each off-diagonal element (indexed using top/bottom pairs), calculate the Jacobi rotation and apply it to Acpy
        for(rocblas_int k = 0; k < even_n - 1; ++k)
        {
            for(rocblas_int cc = 0; cc < count; ++cc)
            {
                // get current top/bottom pair
                rocblas_int kx = tix + cc * dimx;
                i = kx < half_n ? top[kx] : n;
                j = kx < half_n ? bottom[kx] : n;

                // calculate current rotation J
                if(tiy == 0 && i < n && j < n)
                {
                    aij = Acpy[i + j * n];
                    mag = std::abs(aij);

                    if(mag * mag < small_num)
                    {
                        c = 1;
                        s1 = 0;
                    }
                    else
                    {
                        g = 2 * mag;
                        f = std::real(Acpy[j + j * n] - Acpy[i + i * n]);
                        f += (f < 0) ? -std::hypot(f, g) : std::hypot(f, g);
                        lartg(f, g, c, s, r);
                        s1 = s * aij / mag;
                    }
                    cosines_res[tix] = c;
                    sines_diag[tix] = s1;
                }
                __syncthreads();

                // apply J from the right and update vectors
                if(i < n && j < n)
                {
                    c = cosines_res[tix];
                    s1 = sines_diag[tix];
                    s2 = conj(s1);

                    for(rocblas_int ky = tiy; ky < half_n; ky += dimy)
                    {
                        rocblas_int y1 = ky * 2;
                        rocblas_int y2 = y1 + 1;

                        temp1 = Acpy[y1 + i * n];
                        temp2 = Acpy[y1 + j * n];
                        Acpy[y1 + i * n] = c * temp1 + s2 * temp2;
                        Acpy[y1 + j * n] = -s1 * temp1 + c * temp2;
                        if(y2 < n)
                        {
                            temp1 = Acpy[y2 + i * n];
                            temp2 = Acpy[y2 + j * n];
                            Acpy[y2 + i * n] = c * temp1 + s2 * temp2;
                            Acpy[y2 + j * n] = -s1 * temp1 + c * temp2;
                        }

                        if(evect != rocblas_evect_none)
                        {
                            temp1 = A[y1 + i * lda];
                            temp2 = A[y1 + j * lda];
                            A[y1 + i * lda] = c * temp1 + s2 * temp2;
                            A[y1 + j * lda] = -s1 * temp1 + c * temp2;
                            if(y2 < n)
                            {
                                temp1 = A[y2 + i * lda];
                                temp2 = A[y2 + j * lda];
                                A[y2 + i * lda] = c * temp1 + s2 * temp2;
                                A[y2 + j * lda] = -s1 * temp1 + c * temp2;
                            }
                        }
                    }
                }
                __syncthreads();

                // apply J' from the left
                if(i < n && j < n)
                {
                    for(rocblas_int ky = tiy; ky < half_n; ky += dimy)
                    {
                        rocblas_int y1 = ky * 2;
                        rocblas_int y2 = y1 + 1;

                        temp1 = Acpy[i + y1 * n];
                        temp2 = Acpy[j + y1 * n];
                        Acpy[i + y1 * n] = c * temp1 + s1 * temp2;
                        Acpy[j + y1 * n] = -s2 * temp1 + c * temp2;
                        if(y2 < n)
                        {
                            temp1 = Acpy[i + y2 * n];
                            temp2 = Acpy[j + y2 * n];
                            Acpy[i + y2 * n] = c * temp1 + s1 * temp2;
                            Acpy[j + y2 * n] = -s2 * temp1 + c * temp2;
                        }
                    }
                }
                __syncthreads();

                // round aij and aji to zero
                if(tiy == 0 && i < n && j < n)
                {
                    Acpy[i + j * n] = 0;
                    Acpy[j + i * n] = 0;
                }
                __syncthreads();

                // rotate top/bottom pair
                if(tiy == 0 && kx < half_n)
                {
                    if(i > 0)
                    {
                        if(i == 2 || i == even_n - 1)
                            top[kx] = i - 1;
                        else
                            top[kx] = i + ((i % 2 == 0) ? -2 : 2);
                    }
                    if(j == 2 || j == even_n - 1)
                        bottom[kx] = j - 1;
                    else
                        bottom[kx] = j + ((j % 2 == 0) ? -2 : 2);
                }
                __syncthreads();
            }
        }

        // update norm
        if(tiy == 0)
        {
            local_res = 0;

            for(i = tix; i < n; i += dimx)
            {
                for(j = 0; j < i; j++)
                    local_res += 2 * std::norm(Acpy[i + j * n]);
            }
            cosines_res[tix] = local_res;
        }
        __syncthreads();

        local_res = 0;
        for(i = 0; i < dimx; i++)
            local_res += cosines_res[i];

        sweeps++;
    }

    // finalize outputs
    if(tiy == 0)
    {
        if(tix == 0)
        {
            *residual = sqrt(local_res);
            if(sweeps <= max_sweeps)
            {
                *n_sweeps = sweeps;
                *info = 0;
            }
            else
            {
                *n_sweeps = max_sweeps;
                *info = 1;
            }
        }

        // update W
        for(i = tix; i < n; i += dimx)
            W[i] = std::real(Acpy[i + i * n]);
    }
    __syncthreads();

    // if no sort, then stop
    if(esort == rocblas_esort_none)
        return;

    //otherwise sort eigenvalues and eigenvectors by selection sort
    rocblas_int m;
    S p;
    for(j = 0; j < n - 1; j++)
    {
        m = j;
        p = W[j];
        for(i = j + 1; i < n; i++)
        {
            if(W[i] < p)
            {
                m = i;
                p = W[i];
            }
        }
        __syncthreads();

        if(m != j && tiy == 0)
        {
            if(tix == 0)
            {
                W[m] = W[j];
                W[j] = p;
            }

            if(evect != rocblas_evect_none)
            {
                for(i = tix; i < n; i += dimx)
                    swap(A[i + m * lda], A[i + j * lda]);
            }
        }
        __syncthreads();
    }
}
#endif

__host__ __device__ inline void
    rsyevj_get_dims(rocblas_int n, rocblas_int bdim, rocblas_int* ddx, rocblas_int* ddy)
{
    // (TODO: Some tuning could be beneficial in the future.
    //	For now, we use a max of BDIM = ddx * ddy threads.
    //	ddy is set to min(BDIM/4, ceil(n/2)) and ddx to min(BDIM/ddy, ceil(n/2)).

    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;
    rocblas_int x = 32;
    rocblas_int y = std::min(bdim / x, half_n);
    *ddx = x;
    *ddy = y;
}

template <typename T, typename I, typename S, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(RSYEVJ_BDIM)
    rsyevj_small_kernel(const rocblas_esort esort,
                        const rocblas_evect evect,
                        const rocblas_fill uplo,
                        const I n,
                        U AA,
                        const I shiftA,
                        const I lda,
                        const rocblas_stride strideA,
                        const S abstol,
                        const S eps,
                        S* residualA,
                        const I max_sweeps,
                        I* n_sweepsA,
                        S* WW,
                        const rocblas_stride strideW,
                        I* infoA,
                        T* AcpyA,

                        I batch_count,
                        I* schedule_)
{
    auto const tid = hipThreadIdx_x;
    auto const bid_start = hipBlockIdx_z;
    auto const bid_inc = hipGridDim_z;

    auto const even_n = n + (n % 2);
    auto const half_n = even_n / 2;

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        // array pointers
        T* const dA = load_ptr_batch<T>(AA, bid, shiftA, strideA);
        T* const Acpy = AcpyA + bid * n * n;
        auto const dV = Acpy;

        S* const W = WW + bid * strideW;
        S* const residual = residualA + bid;
        I* const n_sweeps = n_sweepsA + bid;
        I* const info = infoA + bid;

        // get dimensions of 2D thread array
        I ddx = 32;
        I ddy = RSYEVJ_BDIM / ddx;
        rsyevj_get_dims(n, RSYEVJ_BDIM, &ddx, &ddy);

        bool const need_V = (esort != rocblas_esort_none);

        // shared memory
        auto const ntables = half_n;
        extern __shared__ double lmem[];
        S* cosines_res = reinterpret_cast<S*>(lmem);
        T* sines_diag = reinterpret_cast<T*>(cosines_res + ntables);
        T* pfree = reinterpret_cast<I*>(sines_diag + ntables);
        T* const A_ = pfree;
        pfree += n * n;
        T* const V_ = (need_V) ? pfree : nullptr;

        // extra check whether to use A_ and V_ in LDS
        {
            auto const max_lds = 64 * 1024;
            size_t total_size = 0;
            total_size += sizeof(S) * n; // cosine_res
            total_size += sizeof(T) * n; // sines_diag
            total_size += sizeof(T) * n * n; // A_
            total_size += (need_V) ? sizeof(T) * n * n : 0; // V_

            bool const can_use_lds = (total_size <= max_lds);
            if(!can_use_lds)
            {
                // ----------------------------
                // need to use GPU device memory
                // ----------------------------
                A_ = dA;
                V_ = (need_V) ? dV : nullptr;
            }

            // ---------------------------------------------
            // check cosine and sine arrays still fit in LDS
            // ---------------------------------------------
            assert((sizeof(S) + sizeof(T)) * ntables <= max_lds);
        }

        // re-arrange threads in 2D array
        I const tix = tid % ddx;
        I const tiy = tid / ddx;

        // execute
        run_rsyevj(ddx, ddy, tix, tiy, esort, evect, uplo, n, dA, lda, abstol, eps, residual,
                   max_sweeps, n_sweeps, W, info, A_, V_, cosines_res, sines_diag, schedule_);

        __syncthreads();

        // ------------------------------
        // over-write original matrix dA
        // with V if eigen-vectors are requested
        // ------------------------------

        if(need_V)
        {
            char const c_uplo = 'A';
            auto const mm = n;
            auto const nn = n;

            auto const ldv = n;
            auto const ld1 = ldv;
            auto const ld2 = lda;

            auto const i_start = tix;
            auto const i_inc = ddx;
            auto const j_start = tiy;
            auto const j_inc = ddy;

            lacpy_body(c_uplo, mm, nn, V_, ld1, dA, ld2, i_start, i_inc, j_start, j_inc);
        }

        __syncthreads();
    }
}

/************** Kernels and device functions for large size*******************/
/*****************************************************************************/

#if(0)
/** SYEVJ_INIT copies A to Acpy, calculates the residual norm of the matrix, and
    initializes the top/bottom pairs.

    Call this kernel with batch_count groups in y, and any number of threads in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_init(const rocblas_evect evect,
                                 const rocblas_fill uplo,
                                 const rocblas_int half_blocks,
                                 const rocblas_int n,
                                 U AA,
                                 const rocblas_int shiftA,
                                 const rocblas_int lda,
                                 const rocblas_stride strideA,
                                 S abstol,
                                 S* residual,
                                 T* AcpyA,
                                 S* norms,
                                 rocblas_int* top,
                                 rocblas_int* bottom,
                                 rocblas_int* completed)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int dimx = hipBlockDim_x;

    // local variables
    T temp;
    rocblas_int i, j;
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* Acpy = AcpyA + bid * n * n;

    // shared memory
    extern __shared__ double lmem[];
    S* sh_res = reinterpret_cast<S*>(lmem);
    S* sh_diag = sh_res + dimx;

    // copy A to Acpy, set A to identity (if calculating eigenvectors), and calculate off-diagonal
    // squared Frobenius norm (by column/row)
    S local_res = 0;
    S local_diag = 0;
    if(uplo == rocblas_fill_upper)
    {
        for(i = tid; i < n; i += dimx)
        {
            temp = A[i + i * lda];
            local_diag += std::norm(temp);
            Acpy[i + i * n] = temp;

            if(evect != rocblas_evect_none)
                A[i + i * lda] = 1;

            for(j = n - 1; j > i; j--)
            {
                temp = A[i + j * lda];
                local_res += 2 * std::norm(temp);
                Acpy[i + j * n] = temp;
                Acpy[j + i * n] = conj(temp);

                if(evect != rocblas_evect_none)
                {
                    A[i + j * lda] = 0;
                    A[j + i * lda] = 0;
                }
            }
        }
    }
    else
    {
        for(i = tid; i < n; i += dimx)
        {
            temp = A[i + i * lda];
            local_diag += std::norm(temp);
            Acpy[i + i * n] = temp;

            if(evect != rocblas_evect_none)
                A[i + i * lda] = 1;

            for(j = 0; j < i; j++)
            {
                temp = A[i + j * lda];
                local_res += 2 * std::norm(temp);
                Acpy[i + j * n] = temp;
                Acpy[j + i * n] = conj(temp);

                if(evect != rocblas_evect_none)
                {
                    A[i + j * lda] = 0;
                    A[j + i * lda] = 0;
                }
            }
        }
    }
    sh_res[tid] = local_res;
    sh_diag[tid] = local_diag;
    __syncthreads();

    if(tid == 0)
    {
        for(i = 1; i < std::min(n, dimx); i++)
        {
            local_res += sh_res[i];
            local_diag += sh_diag[i];
        }

        norms[bid] = (local_res + local_diag) * abstol * abstol;
        residual[bid] = local_res;
        if(local_res < norms[bid])
        {
            completed[bid + 1] = 1;
            atomicAdd(completed, 1);
        }
    }

    // initialize top/bottom pairs
    if(bid == 0 && top && bottom)
    {
        for(i = tid; i < half_blocks; i += dimx)
        {
            top[i] = 2 * i;
            bottom[i] = 2 * i + 1;
        }
    }
}
#endif

/** SYEVJ_DIAG_KERNEL decomposes diagonal blocks of size nb <= BS2. For each off-diagonal element
    A[i,j], a Jacobi rotation J is calculated so that (J'AJ)[i,j] = 0. J only affects rows i and j,
    and J' only affects columns i and j. Therefore, ceil(nb / 2) rotations can be computed and applied
    in parallel, so long as the rotations do not conflict between threads. We use top/bottom pairs
    to obtain i's and j's that do not conflict, and cycle them to cover all off-diagonal indices.

    Call this kernel with batch_count blocks in z, and BS2 / 2 threads in x and y. Each thread block
    will work on a separate diagonal block; for a matrix consisting of b * b blocks, use b thread
    blocks in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void rsyevj_diag_kernel(const rocblas_int n,
                                         U AA,
                                         const rocblas_int shiftA,
                                         const rocblas_int lda,
                                         const rocblas_stride strideA,
                                         const S eps,
                                         T* JA,
                                         rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + hipBlockIdx_x;

    if(completed[bid + 1])
        return;

    rocblas_int nb_max = 2 * hipBlockDim_x;
    rocblas_int offset = hipBlockIdx_x * nb_max;

    // local variables
    S c, mag, f, g, r, s;
    T s1, s2, aij, temp1, temp2;
    rocblas_int i, j, k;
    rocblas_int xx1 = 2 * tix, xx2 = xx1 + 1;
    rocblas_int yy1 = 2 * tiy, yy2 = yy1 + 1;
    rocblas_int x1 = xx1 + offset, x2 = x1 + 1;
    rocblas_int y1 = yy1 + offset, y2 = y1 + 1;

    rocblas_int half_n = (n - 1) / 2 + 1;
    rocblas_int nb = std::min(2 * half_n - offset, nb_max);
    rocblas_int half_nb = nb / 2;

    if(tix >= half_nb || tiy >= half_nb)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = (JA ? JA + (jid * nb_max * nb_max) : nullptr);

    // shared memory
    extern __shared__ double lmem[];
    S* sh_cosines = reinterpret_cast<S*>(lmem);
    T* sh_sines = reinterpret_cast<T*>(sh_cosines + half_nb);
    rocblas_int* sh_top = reinterpret_cast<rocblas_int*>(sh_sines + half_nb);
    rocblas_int* sh_bottom = sh_top + half_nb;

    // initialize J to the identity
    if(J)
    {
        J[xx1 + yy1 * nb_max] = (xx1 == yy1 ? 1 : 0);
        J[xx1 + yy2 * nb_max] = 0;
        J[xx2 + yy1 * nb_max] = 0;
        J[xx2 + yy2 * nb_max] = (xx2 == yy2 ? 1 : 0);
    }

    // initialize top/bottom
    if(tiy == 0)
    {
        sh_top[tix] = x1;
        sh_bottom[tix] = x2;
    }

    S small_num = get_safemin<S>() / eps;

    // for each off-diagonal element (indexed using top/bottom pairs), calculate the Jacobi rotation and apply it to A
    i = x1;
    j = x2;
    for(k = 0; k < nb - 1; k++)
    {
        if(tiy == 0 && i < n && j < n)
        {
            aij = A[i + j * lda];
            mag = std::abs(aij);

            // calculate rotation J
            if(mag * mag < small_num)
            {
                c = 1;
                s1 = 0;
            }
            else
            {
                g = 2 * mag;
                f = std::real(A[j + j * lda] - A[i + i * lda]);
                f += (f < 0) ? -std::hypot(f, g) : std::hypot(f, g);
                lartg(f, g, c, s, r);
                s1 = s * aij / mag;
            }

            sh_cosines[tix] = c;
            sh_sines[tix] = s1;
        }
        __syncthreads();

        if(i < n && j < n)
        {
            c = sh_cosines[tix];
            s1 = sh_sines[tix];
            s2 = conj(s1);

            // store J row-wise
            if(J)
            {
                xx1 = i - offset;
                xx2 = j - offset;
                temp1 = J[xx1 + yy1 * nb_max];
                temp2 = J[xx2 + yy1 * nb_max];
                J[xx1 + yy1 * nb_max] = c * temp1 + s2 * temp2;
                J[xx2 + yy1 * nb_max] = -s1 * temp1 + c * temp2;

                if(y2 < n)
                {
                    temp1 = J[xx1 + yy2 * nb_max];
                    temp2 = J[xx2 + yy2 * nb_max];
                    J[xx1 + yy2 * nb_max] = c * temp1 + s2 * temp2;
                    J[xx2 + yy2 * nb_max] = -s1 * temp1 + c * temp2;
                }
            }

            // apply J from the right
            temp1 = A[y1 + i * lda];
            temp2 = A[y1 + j * lda];
            A[y1 + i * lda] = c * temp1 + s2 * temp2;
            A[y1 + j * lda] = -s1 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[y2 + i * lda];
                temp2 = A[y2 + j * lda];
                A[y2 + i * lda] = c * temp1 + s2 * temp2;
                A[y2 + j * lda] = -s1 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(i < n && j < n)
        {
            // apply J' from the left
            temp1 = A[i + y1 * lda];
            temp2 = A[j + y1 * lda];
            A[i + y1 * lda] = c * temp1 + s1 * temp2;
            A[j + y1 * lda] = -s2 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[i + y2 * lda];
                temp2 = A[j + y2 * lda];
                A[i + y2 * lda] = c * temp1 + s1 * temp2;
                A[j + y2 * lda] = -s2 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(tiy == 0 && i < n && j < n)
        {
            // round aij and aji to zero
            A[i + j * lda] = 0;
            A[j + i * lda] = 0;
        }

        // cycle top/bottom pairs
        if(tix == 1)
            i = sh_bottom[0];
        else if(tix > 1)
            i = sh_top[tix - 1];
        if(tix == half_nb - 1)
            j = sh_top[half_nb - 1];
        else
            j = sh_bottom[tix + 1];
        __syncthreads();

        if(tiy == 0)
        {
            sh_top[tix] = i;
            sh_bottom[tix] = j;
        }
    }
}

/** SYEVJ_DIAG_ROTATE rotates off-diagonal blocks of size nb <= BS2 using the rotations calculated
    by SYEVJ_DIAG_KERNEL.

    Call this kernel with batch_count groups in z, and BS2 threads in x and y. Each thread group
    will work on a separate off-diagonal block; for a matrix consisting of b * b blocks, use b groups
    in x and b - 1 groups in y. **/
template <bool APPLY_LEFT, typename T, typename S, typename U>
ROCSOLVER_KERNEL void rsyevj_diag_rotate(const bool skip_block,
                                         const rocblas_int n,
                                         U AA,
                                         const rocblas_int shiftA,
                                         const rocblas_int lda,
                                         const rocblas_stride strideA,
                                         T* JA,
                                         rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bix = hipBlockIdx_x;
    rocblas_int biy = hipBlockIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + bix;

    if(completed[bid + 1])
        return;
    if(skip_block && bix == biy)
        return;

    rocblas_int nb_max = hipBlockDim_x;
    rocblas_int offsetx = bix * nb_max;
    rocblas_int offsety = biy * nb_max;

    // local variables
    T temp;
    rocblas_int k;
    rocblas_int x = tix + offsetx;
    rocblas_int y = tiy + offsety;

    rocblas_int nb = std::min(n - offsetx, nb_max);

    if(x >= n || y >= n)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = JA + (jid * nb_max * nb_max);

    // apply J to the current block
    if(!APPLY_LEFT)
    {
        temp = 0;
        for(k = 0; k < nb; k++)
            temp += J[tix + k * nb_max] * A[y + (k + offsetx) * lda];
        __syncthreads();
        A[y + x * lda] = temp;
    }
    else
    {
        temp = 0;
        for(k = 0; k < nb; k++)
            temp += conj(J[tix + k * nb_max]) * A[(k + offsetx) + y * lda];
        __syncthreads();
        A[x + y * lda] = temp;
    }
}

#if(0)
/** SYEVJ_OFFD_KERNEL decomposes off-diagonal blocks of size nb <= BS2. For each element in the block
    (which is an off-diagonal element A[i,j] in the matrix A), a Jacobi rotation J is calculated so that
    (J'AJ)[i,j] = 0. J only affects rows i and j, and J' only affects columns i and j. Therefore,
    nb rotations can be computed and applied in parallel, so long as the rotations do not conflict between
    threads. We select the initial set of i's and j's to span the block's diagonal, and iteratively move
    to the right (wrapping around as necessary) to cover all indices.

    Since A[i,i], A[j,j], and A[j,i] are all in separate blocks, we also need to ensure that
    rotations do not conflict between thread groups. We use block-level top/bottom pairs
    to obtain off-diagonal block indices that do not conflict.

    Call this kernel with batch_count groups in z, and BS2 threads in x and y. Each thread group
    will work on four matrix blocks; for a matrix consisting of b * b blocks, use b / 2 groups in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void rsyevj_offd_kernel(const rocblas_int blocks,
                                         const rocblas_int n,
                                         U AA,
                                         const rocblas_int shiftA,
                                         const rocblas_int lda,
                                         const rocblas_stride strideA,
                                         const S eps,
                                         T* JA,
                                         rocblas_int* top,
                                         rocblas_int* bottom,
                                         rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + hipBlockIdx_x;

    if(completed[bid + 1])
        return;

    rocblas_int i = top[hipBlockIdx_x];
    rocblas_int j = bottom[hipBlockIdx_x];
    if(i >= blocks || j >= blocks)
        return;
    if(i > j)
        swap(i, j);

    rocblas_int nb_max = hipBlockDim_x;
    rocblas_int offseti = i * nb_max;
    rocblas_int offsetj = j * nb_max;
    rocblas_int ldj = 2 * nb_max;

    // local variables
    S c, mag, f, g, r, s;
    T s1, s2, aij, temp1, temp2;
    rocblas_int k;
    rocblas_int xx1 = tix, xx2 = tix + nb_max;
    rocblas_int yy1 = tiy, yy2 = tiy + nb_max;
    rocblas_int x1 = tix + offseti, x2 = tix + offsetj;
    rocblas_int y1 = tiy + offseti, y2 = tiy + offsetj;

    if(y1 >= n)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = (JA ? JA + (jid * 4 * nb_max * nb_max) : nullptr);

    // shared memory
    extern __shared__ double lmem[];
    S* sh_cosines = reinterpret_cast<S*>(lmem);
    T* sh_sines = reinterpret_cast<T*>(sh_cosines + nb_max);

    // initialize J to the identity
    if(J)
    {
        J[xx1 + yy1 * ldj] = (xx1 == yy1 ? 1 : 0);
        J[xx1 + yy2 * ldj] = 0;
        J[xx2 + yy1 * ldj] = 0;
        J[xx2 + yy2 * ldj] = (xx2 == yy2 ? 1 : 0);
    }

    S small_num = get_safemin<S>() / eps;

    // for each element, calculate the Jacobi rotation and apply it to A
    for(k = 0; k < nb_max; k++)
    {
        // get element indices
        i = x1;
        j = (tix + k) % nb_max + offsetj;

        if(tiy == 0 && i < n && j < n)
        {
            aij = A[i + j * lda];
            mag = std::abs(aij);

            // calculate rotation J
            if(mag * mag < small_num)
            {
                c = 1;
                s1 = 0;
            }
            else
            {
                g = 2 * mag;
                f = std::real(A[j + j * lda] - A[i + i * lda]);
                f += (f < 0) ? -std::hypot(f, g) : std::hypot(f, g);
                lartg(f, g, c, s, r);
                s1 = s * aij / mag;
            }

            sh_cosines[tix] = c;
            sh_sines[tix] = s1;
        }
        __syncthreads();

        if(i < n && j < n)
        {
            c = sh_cosines[tix];
            s1 = sh_sines[tix];
            s2 = conj(s1);

            // store J row-wise
            if(J)
            {
                xx1 = i - offseti;
                xx2 = j - offsetj + nb_max;
                temp1 = J[xx1 + yy1 * ldj];
                temp2 = J[xx2 + yy1 * ldj];
                J[xx1 + yy1 * ldj] = c * temp1 + s2 * temp2;
                J[xx2 + yy1 * ldj] = -s1 * temp1 + c * temp2;

                if(y2 < n)
                {
                    temp1 = J[xx1 + yy2 * ldj];
                    temp2 = J[xx2 + yy2 * ldj];
                    J[xx1 + yy2 * ldj] = c * temp1 + s2 * temp2;
                    J[xx2 + yy2 * ldj] = -s1 * temp1 + c * temp2;
                }
            }

            // apply J from the right
            temp1 = A[y1 + i * lda];
            temp2 = A[y1 + j * lda];
            A[y1 + i * lda] = c * temp1 + s2 * temp2;
            A[y1 + j * lda] = -s1 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[y2 + i * lda];
                temp2 = A[y2 + j * lda];
                A[y2 + i * lda] = c * temp1 + s2 * temp2;
                A[y2 + j * lda] = -s1 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(i < n && j < n)
        {
            // apply J' from the left
            temp1 = A[i + y1 * lda];
            temp2 = A[j + y1 * lda];
            A[i + y1 * lda] = c * temp1 + s1 * temp2;
            A[j + y1 * lda] = -s2 * temp1 + c * temp2;

            if(y2 < n)
            {
                temp1 = A[i + y2 * lda];
                temp2 = A[j + y2 * lda];
                A[i + y2 * lda] = c * temp1 + s1 * temp2;
                A[j + y2 * lda] = -s2 * temp1 + c * temp2;
            }
        }
        __syncthreads();

        if(tiy == 0 && j < n)
        {
            // round aij and aji to zero
            A[i + j * lda] = 0;
            A[j + i * lda] = 0;
        }
    }
}
#endif

#if(0)
/** SYEVJ_OFFD_ROTATE rotates off-diagonal blocks using the rotations calculated by SYEVJ_OFFD_KERNEL.

    Call this kernel with batch_count groups in z, 2*BS2 threads in x and BS2/2 threads in y.
    For a matrix consisting of b * b blocks, use b / 2 groups in x and 2(b - 2) groups in y. **/
template <bool APPLY_LEFT, typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_offd_rotate(const bool skip_block,
                                        const rocblas_int blocks,
                                        const rocblas_int n,
                                        U AA,
                                        const rocblas_int shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        T* JA,
                                        rocblas_int* top,
                                        rocblas_int* bottom,
                                        rocblas_int* completed)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int tiy = hipThreadIdx_y;
    rocblas_int bix = hipBlockIdx_x;
    rocblas_int biy = hipBlockIdx_y;
    rocblas_int bid = hipBlockIdx_z;
    rocblas_int jid = bid * hipGridDim_x + hipBlockIdx_x;

    if(completed[bid + 1])
        return;

    rocblas_int i = top[bix];
    rocblas_int j = bottom[bix];
    if(i >= blocks || j >= blocks)
        return;
    if(i > j)
        swap(i, j);
    if(skip_block && (biy / 2 == i || biy / 2 == j))
        return;

    rocblas_int nb_max = hipBlockDim_x / 2;
    rocblas_int offseti = i * nb_max;
    rocblas_int offsetj = j * nb_max;
    rocblas_int offsetx = (tix < nb_max ? offseti : offsetj - nb_max);
    rocblas_int offsety = biy * hipBlockDim_y;
    rocblas_int ldj = 2 * nb_max;

    // local variables
    T temp;
    rocblas_int k;
    rocblas_int x = tix + offsetx;
    rocblas_int y = tiy + offsety;

    rocblas_int nb = std::min(n - offsetj, nb_max);

    if(x >= n || y >= n)
        return;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    T* J = JA + (jid * 4 * nb_max * nb_max);

    // apply J to the current block
    if(!APPLY_LEFT)
    {
        temp = 0;
        for(k = 0; k < nb_max; k++)
            temp += J[tix + k * ldj] * A[y + (k + offseti) * lda];
        for(k = 0; k < nb; k++)
            temp += J[tix + (k + nb_max) * ldj] * A[y + (k + offsetj) * lda];
        __syncthreads();
        A[y + x * lda] = temp;
    }
    else
    {
        temp = 0;
        for(k = 0; k < nb_max; k++)
            temp += conj(J[tix + k * ldj]) * A[(k + offseti) + y * lda];
        for(k = 0; k < nb; k++)
            temp += conj(J[tix + (k + nb_max) * ldj]) * A[(k + offsetj) + y * lda];
        __syncthreads();
        A[x + y * lda] = temp;
    }
}
#endif

#if(0)
/** SYEVJ_CYCLE_PAIRS cycles the block-level top/bottom pairs to progress the sweep.

    Call this kernel with any number of threads in x. (Top/bottom pairs are shared across batch instances,
    so only one thread group is needed.) **/
template <typename T>
ROCSOLVER_KERNEL void
    syevj_cycle_pairs(const rocblas_int half_blocks, rocblas_int* top, rocblas_int* bottom)
{
    rocblas_int tix = hipThreadIdx_x;
    rocblas_int i, j, k;

    if(half_blocks <= hipBlockDim_x && tix < half_blocks)
    {
        if(tix == 0)
            i = 0;
        else if(tix == 1)
            i = bottom[0];
        else if(tix > 1)
            i = top[tix - 1];

        if(tix == half_blocks - 1)
            j = top[half_blocks - 1];
        else
            j = bottom[tix + 1];
        __syncthreads();

        top[tix] = i;
        bottom[tix] = j;
    }
    else
    {
        // shared memory
        extern __shared__ double lmem[];
        rocblas_int* sh_top = reinterpret_cast<rocblas_int*>(lmem);
        rocblas_int* sh_bottom = reinterpret_cast<rocblas_int*>(sh_top + half_blocks);

        for(k = tix; k < half_blocks; k += hipBlockDim_x)
        {
            sh_top[k] = top[k];
            sh_bottom[k] = bottom[k];
        }
        __syncthreads();

        for(k = tix; k < half_blocks; k += hipBlockDim_x)
        {
            if(k == 1)
                top[k] = sh_bottom[0];
            else if(k > 1)
                top[k] = sh_top[k - 1];

            if(k == half_blocks - 1)
                bottom[k] = sh_top[half_blocks - 1];
            else
                bottom[k] = sh_bottom[k + 1];
        }
    }
}
#endif

#if(0)
/** SYEVJ_CALC_NORM calculates the residual norm of the matrix.

    Call this kernel with batch_count groups in y, and any number of threads in x. **/
template <typename T, typename S>
ROCSOLVER_KERNEL void syevj_calc_norm(const rocblas_int n,
                                      const rocblas_int sweeps,
                                      S* residual,
                                      T* AcpyA,
                                      S* norms,
                                      rocblas_int* completed)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;
    rocblas_int dimx = hipBlockDim_x;

    if(completed[bid + 1])
        return;

    // local variables
    rocblas_int i, j;

    // array pointers
    T* Acpy = AcpyA + bid * n * n;

    // shared memory
    extern __shared__ double lmem[];
    S* sh_res = reinterpret_cast<S*>(lmem);

    S local_res = 0;
    for(i = tid; i < n; i += dimx)
    {
        for(j = 0; j < i; j++)
            local_res += 2 * std::norm(Acpy[i + j * n]);
    }
    sh_res[tid] = local_res;
    __syncthreads();

    if(tid == 0)
    {
        for(i = 1; i < std::min(n, dimx); i++)
            local_res += sh_res[i];

        residual[bid] = local_res;
        if(local_res < norms[bid])
        {
            completed[bid + 1] = sweeps + 1;
            atomicAdd(completed, 1);
        }
    }
}
#endif

#if(0)
/** SYEVJ_FINALIZE sets the output values for SYEVJ, and sorts the eigenvalues and
    eigenvectors by selection sort if applicable.

    Call this kernel with batch_count groups in y, and any number of threads in x. **/
template <typename T, typename S, typename U>
ROCSOLVER_KERNEL void syevj_finalize(const rocblas_esort esort,
                                     const rocblas_evect evect,
                                     const rocblas_int n,
                                     U AA,
                                     const rocblas_int shiftA,
                                     const rocblas_int lda,
                                     const rocblas_stride strideA,
                                     S* residual,
                                     const rocblas_int max_sweeps,
                                     rocblas_int* n_sweeps,
                                     S* WW,
                                     const rocblas_stride strideW,
                                     rocblas_int* info,
                                     T* AcpyA,
                                     rocblas_int* completed)
{
    rocblas_int tid = hipThreadIdx_x;
    rocblas_int bid = hipBlockIdx_y;

    // local variables
    rocblas_int i, j, m;
    rocblas_int sweeps = 0;

    // array pointers
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    S* W = WW + bid * strideW;
    T* Acpy = AcpyA + bid * n * n;

    // finalize outputs
    if(tid == 0)
    {
        rocblas_int sweeps = completed[bid + 1] - 1;
        residual[bid] = sqrt(residual[bid]);
        if(sweeps >= 0)
        {
            n_sweeps[bid] = sweeps;
            info[bid] = 0;
        }
        else
        {
            n_sweeps[bid] = max_sweeps;
            info[bid] = 1;
        }
    }

    // put eigenvalues into output array
    for(i = tid; i < n; i += hipBlockDim_x)
        W[i] = std::real(Acpy[i + i * n]);
    __syncthreads();

    if((evect == rocblas_evect_none && tid > 0) || esort == rocblas_esort_none)
        return;

    // sort eigenvalues & vectors
    S p;
    for(j = 0; j < n - 1; j++)
    {
        m = j;
        p = W[j];
        for(i = j + 1; i < n; i++)
        {
            if(W[i] < p)
            {
                m = i;
                p = W[i];
            }
        }
        __syncthreads();

        if(m != j)
        {
            if(tid == 0)
            {
                W[m] = W[j];
                W[j] = p;
            }

            if(evect != rocblas_evect_none)
            {
                for(i = tid; i < n; i += hipBlockDim_x)
                    swap(A[i + m * lda], A[i + j * lda]);
                __syncthreads();
            }
        }
    }
}
#endif

/****** Template function, workspace size and argument validation **********/
/***************************************************************************/

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename I, typename S>
void rocsolver_rsyevj_rheevj_getMemorySize(const rocblas_evect evect,
                                           const rocblas_fill uplo,
                                           const I n,
                                           const I batch_count,
                                           size_t* size_Acpy,
                                           size_t* size_J,
                                           size_t* size_norms,
                                           size_t* size_top,
                                           size_t* size_bottom,
                                           size_t* size_completed)
{
    // set workspace to zero
    {
        *size_Acpy = 0;
        *size_J = 0;
        *size_norms = 0;
        *size_top = 0;
        *size_bottom = 0;
        *size_completed = 0;
    }

    // quick return
    if(n <= 1 || batch_count == 0)
    {
        return;
    }

    // size of temporary workspace for copying A
    *size_Acpy = sizeof(T) * n * n * batch_count;

    bool const need_V = (evect == rocblas_evect_none);

    if(n <= RSYEVJ_BLOCKED_SWITCH(T, need_V))
    {
        *size_J = 0;
        *size_norms = 0;
        *size_top = 0;
        *size_bottom = 0;
        *size_completed = 0;
        return;
    }

    auto const ceil(auto n, auto nb)
    {
        return ((n - 1) / nb + 1);
    };
    auto const is_even = [](auto n) { return ((n % 2) == 0); };

    I const nb0 = RSYEVJ_BLOCKED_SWITCH(T, need_V) / 2;
    auto nb = nb0;

    I n_even = n + (n % 2);
    I half_n = n_even / 2;

    I nblocks = ceil(n, nb);
    I nblocks_even = nblocks + (nblocks % 2);

    // ------------------------------------------------------------
    // adjust block size nb so that there are even number of blocks
    //
    // For example, if n = 97, initial nb = 16, then
    // 97/16  = 6.06, so ceil(97,16) = 7
    // want even number of  blocks, so need 8 blocks
    // we have 97/8 = 12.125, so adjust nb = 13
    // note 13 * 7 = 91, and 13 * 8 = 104
    // so n = 97 = 13 * 7 + 6,  so there are 8 blocks but the last
    // block is only partially filled with only 6 rows and 6 columns
    // ------------------------------------------------------------
    nb = ceil(n, nblocks_even);
    nb_even = nb + (nb % 2);
    assert((nb <= nb0));

    nblocks = ceil(n, nb);
    nblocks_even = nblocks + (nblocks % 2);
    assert(is_even(nblocks));

    // size of temporary workspace to store the block rotation matrices
    *size_J = (sizeof(T) * (2 * nb) * (2 * nb)) * (nblocks / 2) * batch_count;

    // size of copy of eigen vectors
    if(need_V)
    {
        *size_J += (sizeof(T) * n * n) * batch_count;
    }

    // size of temporary workspace to store the full matrix norm
    *size_norms = sizeof(S) * batch_count;

    // --------------------------------------------------
    // size of arrays for top/bottom tournament schedules
    // to look up independent pairs (p,q) for use in
    // Jacobi method
    // --------------------------------------------------
    *size_top = nblocks_even * (nblocks_even - 1);
    *size_bottom = (2 * nb) * ((2 * nb) - 1);

    // size of temporary workspace to indicate problem completion
    *size_completed = sizeof(I) * (batch_count + 1);
}

#if(0)
/** Argument checking **/
template <typename T, typename S>
rocblas_status rocsolver_syevj_heevj_argCheck(rocblas_handle handle,
                                              const rocblas_esort esort,
                                              const rocblas_evect evect,
                                              const rocblas_fill uplo,
                                              const rocblas_int n,
                                              T A,
                                              const rocblas_int lda,
                                              S* residual,
                                              const rocblas_int max_sweeps,
                                              rocblas_int* n_sweeps,
                                              S* W,
                                              rocblas_int* info,
                                              const rocblas_int batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(esort != rocblas_esort_none && esort != rocblas_esort_ascending)
        return rocblas_status_invalid_value;
    if((evect != rocblas_evect_original && evect != rocblas_evect_none)
       || (uplo != rocblas_fill_lower && uplo != rocblas_fill_upper))
        return rocblas_status_invalid_value;

    // 2. invalid size
    if(n < 0 || lda < n || max_sweeps <= 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((n && !A) || (n && !W) || (batch_count && !residual) || (batch_count && !n_sweeps)
       || (batch_count && !info))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}
#endif

#if(0)
template <bool BATCHED, bool STRIDED, typename T, typename S, typename U>
rocblas_status rocsolver_rsyevj_rheevj_template(rocblas_handle handle,
                                                const rocblas_esort esort,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const rocblas_int n,
                                                U A,
                                                const rocblas_int shiftA,
                                                const rocblas_int lda,
                                                const rocblas_stride strideA,
                                                const S abstol,
                                                S* residual,
                                                const rocblas_int max_sweeps,
                                                rocblas_int* n_sweeps,
                                                S* W,
                                                const rocblas_stride strideW,
                                                rocblas_int* info,
                                                const rocblas_int batch_count,
                                                T* Acpy,
                                                T* J,
                                                S* norms,
                                                rocblas_int* top,
                                                rocblas_int* bottom,
                                                rocblas_int* completed)
{
    ROCSOLVER_ENTER("syevj_heevj", "esort:", esort, "evect:", evect, "uplo:", uplo, "n:", n,
                    "shiftA:", shiftA, "lda:", lda, "abstol:", abstol, "max_sweeps:", max_sweeps,
                    "bc:", batch_count);

    // quick return
    if(batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // quick return
    if(n <= 1)
    {
        rocblas_int blocksReset = (batch_count - 1) / BS1 + 1;
        dim3 gridReset(blocksReset, 1, 1);
        dim3 threadsReset(BS1, 1, 1);

        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, residual,
                                batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, n_sweeps,
                                batch_count, 0);
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

        // scalar case
        if(n == 1)
            ROCSOLVER_LAUNCH_KERNEL(scalar_case<T>, gridReset, threadsReset, 0, stream, evect, A,
                                    strideA, W, strideW, batch_count);

        return rocblas_status_success;
    }

    // absolute tolerance for evaluating when the algorithm has converged
    S eps = get_epsilon<S>();
    S atol = (abstol <= 0 ? eps : abstol);

    // local variables
    rocblas_int even_n = n + n % 2;
    rocblas_int half_n = even_n / 2;
    bool const need_V = (evect != rocblas_evect_none);

    if(n <= RSYEVJ_BLOCKED_SWITCH(T, need_V))
    {
        // *** USE SINGLE SMALL-SIZE KERNEL ***
        // (TODO: SYEVJ_BLOCKED_SWITCH may need re-tuning as it could be larger than 64 now).
        dim3 grid(1, 1, batch_count);

        rocblas_int ddx = 32;
        rocblas_int ddy = 32;

        constexpr bool use_rsyevj = true;
        if(use_rsyevj)
        {
            rsyevj_get_dims(n, RSYEVJ_BDIM, &ddx, &ddy);
            dim3 threads(ddx, ddy, 1);

            size_t const size_V = (need_V) ? sizeof(T) * even_n * even_n : 0;
            size_t const size_A = sizeof(T) * even_n * even_n;
            size_t lmemsize = (sizeof(S) + sizeof(T)) * even_n + size_A + size_V;

            std::vector<rocblas_int> h_schedule(n_even * (n_even - 1));

            ROCSOLVER_LAUNCH_KERNEL(syevj_small_kernel<T>, grid, threads, lmemsize, stream, esort,
                                    evect, uplo, n, A, shiftA, lda, strideA, atol, eps, residual,
                                    max_sweeps, n_sweeps, W, strideW, info, Acpy, d_schedule);
        }
        else
        {
            syevj_get_dims(n, RSYEVJ_BDIM, &ddx, &ddy);
            dim3 threads(ddx * ddy, 1, 1);
            size_t lmemsize = (sizeof(S) + sizeof(T)) * ddx + 2 * sizeof(rocblas_int) * half_n;

            ROCSOLVER_LAUNCH_KERNEL(syevj_small_kernel<T>, grid, threads, lmemsize, stream, esort,
                                    evect, uplo, n, A, shiftA, lda, strideA, atol, eps, residual,
                                    max_sweeps, n_sweeps, W, strideW, info, Acpy);
        }
    }
    else
    {
        // *** USE BLOCKED KERNELS ***

        // kernel dimensions
        rocblas_int blocksReset = batch_count / BS1 + 1;
        rocblas_int blocks = (n - 1) / BS2 + 1;
        rocblas_int even_blocks = blocks + blocks % 2;
        rocblas_int half_blocks = even_blocks / 2;

        dim3 gridReset(blocksReset, 1, 1);
        dim3 grid(1, batch_count, 1);
        dim3 gridDK(blocks, 1, batch_count);
        dim3 gridDR(blocks, blocks, batch_count);
        dim3 gridOK(half_blocks, 1, batch_count);
        dim3 gridOR(half_blocks, 2 * blocks, batch_count);
        dim3 gridPairs(1, 1, 1);
        dim3 threadsReset(BS1, 1, 1);
        dim3 threads(BS1, 1, 1);
        dim3 threadsDK(BS2 / 2, BS2 / 2, 1);
        dim3 threadsDR(BS2, BS2, 1);
        dim3 threadsOK(BS2, BS2, 1);
        dim3 threadsOR(2 * BS2, BS2 / 2, 1);

        // shared memory sizes
        size_t lmemsizeInit = 2 * sizeof(S) * BS1;
        size_t lmemsizeDK = (sizeof(S) + sizeof(T) + 2 * sizeof(rocblas_int)) * (BS2 / 2);
        size_t lmemsizeDR = (2 * sizeof(rocblas_int)) * (BS2 / 2);
        size_t lmemsizeOK = (sizeof(S) + sizeof(T)) * BS2;
        size_t lmemsizePairs = (half_blocks > BS1 ? 2 * sizeof(rocblas_int) * half_blocks : 0);

        bool ev = (evect != rocblas_evect_none);
        rocblas_int h_sweeps = 0;
        rocblas_int h_completed = 0;

        // set completed = 0
        ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, completed,
                                batch_count + 1, 0);

        // copy A to Acpy, set A to identity (if applicable), compute initial residual, and
        // initialize top/bottom pairs (if applicable)
        ROCSOLVER_LAUNCH_KERNEL(syevj_init<T>, grid, threads, lmemsizeInit, stream, evect, uplo,
                                half_blocks, n, A, shiftA, lda, strideA, atol, residual, Acpy,
                                norms, top, bottom, completed);

        while(h_sweeps < max_sweeps)
        {
            // if all instances in the batch have finished, exit the loop
            HIP_CHECK(hipMemcpyAsync(&h_completed, completed, sizeof(rocblas_int),
                                     hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            if(h_completed == batch_count)
                break;

            // decompose diagonal blocks
            ROCSOLVER_LAUNCH_KERNEL(syevj_diag_kernel<T>, gridDK, threadsDK, lmemsizeDK, stream, n,
                                    Acpy, 0, n, n * n, eps, J, completed);

            // apply rotations calculated by diag_kernel
            ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<false, T, S>), gridDR, threadsDR, lmemsizeDR,
                                    stream, true, n, Acpy, 0, n, n * n, J, completed);
            ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<true, T, S>), gridDR, threadsDR, lmemsizeDR,
                                    stream, true, n, Acpy, 0, n, n * n, J, completed);

            // update eigenvectors
            if(ev)
                ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<false, T, S>), gridDR, threadsDR,
                                        lmemsizeDR, stream, false, n, A, shiftA, lda, strideA, J,
                                        completed);

            if(half_blocks == 1)
            {
                // decompose off-diagonal block
                ROCSOLVER_LAUNCH_KERNEL((syevj_offd_kernel<T, S>), gridOK, threadsOK, lmemsizeOK,
                                        stream, blocks, n, Acpy, 0, n, n * n, eps,
                                        (ev ? J : nullptr), top, bottom, completed);

                // update eigenvectors
                if(ev)
                    ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR, threadsOR, 0,
                                            stream, false, blocks, n, A, shiftA, lda, strideA, J,
                                            top, bottom, completed);
            }
            else
            {
                for(rocblas_int b = 0; b < even_blocks - 1; b++)
                {
                    // decompose off-diagonal blocks, indexed by top/bottom pairs
                    ROCSOLVER_LAUNCH_KERNEL((syevj_offd_kernel<T, S>), gridOK, threadsOK,
                                            lmemsizeOK, stream, blocks, n, Acpy, 0, n, n * n, eps,
                                            J, top, bottom, completed);

                    // apply rotations calculated by offd_kernel
                    ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR, threadsOR, 0,
                                            stream, true, blocks, n, Acpy, 0, n, n * n, J, top,
                                            bottom, completed);
                    ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<true, T, S>), gridOR, threadsOR, 0,
                                            stream, true, blocks, n, Acpy, 0, n, n * n, J, top,
                                            bottom, completed);

                    // update eigenvectors
                    if(ev)
                        ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR, threadsOR,
                                                0, stream, false, blocks, n, A, shiftA, lda,
                                                strideA, J, top, bottom, completed);

                    // cycle top/bottom pairs
                    ROCSOLVER_LAUNCH_KERNEL(syevj_cycle_pairs<T>, gridPairs, threads, lmemsizePairs,
                                            stream, half_blocks, top, bottom);
                }
            }

            // compute new residual
            h_sweeps++;
            ROCSOLVER_LAUNCH_KERNEL(syevj_calc_norm<T>, grid, threads, lmemsizeInit, stream, n,
                                    h_sweeps, residual, Acpy, norms, completed);
        }

        // set outputs and sort eigenvalues & vectors
        ROCSOLVER_LAUNCH_KERNEL(syevj_finalize<T>, grid, threads, 0, stream, esort, evect, n, A,
                                shiftA, lda, strideA, residual, max_sweeps, n_sweeps, W, strideW,
                                info, Acpy, completed);
    }

    return rocblas_status_success;
}
#endif

ROCSOLVER_END_NAMESPACE
#undef RSYEVJ_BDIM
