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

/************** CPU functions                              *******************/
/*****************************************************************************/

// -----------------------------------------------------
// CPU code to generate tournament schedule to identity
// the set of independent pairs (p,q) for each round
//
// The schedule is viewed as a "round robin" ping pong tournament
// for n (n is even) players
//
// Each player must play all other (n-1) players
// There are  (n/2) tables and there are (n-1) rounds
// The schedule lists the players for each round
//
// Note if there are odd number of players, say n is odd
// we can generate the schedule for the next higher even number of players,
// n_even = n + (n % 2 ) players
// but skip matches for the non-existant player "n_even"
//
// **NOTE**
// in this implementation, the last player stays fixed at the last table
// in the last position
//
// For example, here is the schedule for 6 players
// It requires 3 tables and 5 rounds
//
// * NOTE * player 5 always plays on the last table Table 2
//
// === round 0 ===
// player 0 against  player 4 on table 0
// player 1 against  player 3 on table 1
// player 2 against  player 5 on table 2
// === round 1 ===
// player 0 against  player 1 on table 0
// player 2 against  player 4 on table 1
// player 3 against  player 5 on table 2
// === round 2 ===
// player 1 against  player 2 on table 0
// player 0 against  player 3 on table 1
// player 4 against  player 5 on table 2
// === round 3 ===
// player 2 against  player 3 on table 0
// player 1 against  player 4 on table 1
// player 0 against  player 5 on table 2
// === round 4 ===
// player 3 against  player 4 on table 0
// player 0 against  player 2 on table 1
// player 1 against  player 5 on table 2
// -----------------------------------------------------

template <typename I>
static void generateTournamentSequence(I const nplayers, std::vector<I>& result)
{
    result.resize(0);

    if(nplayers < 2)
    {
        std::cout << "At least 2 players are needed for a tournament." << std::endl;
        return;
    }

    // ------------------------------------------
    // make sure there are even number of players
    // ------------------------------------------
    auto const m = nplayers + (nplayers % 2);

    // Calculate number of rounds required
    auto const numRounds = m - 1;

    std::vector<I> vec(m);
    std::vector<I> map(m);

    auto rotate = [&]() {
        auto const v0 = vec[0];
        for(I i = 1; i < (m - 1); i++)
        {
            vec[i - 1] = vec[i];
        };
        vec[m - 2] = v0;
    };

    I const ntables = m / 2;
    auto gen_schedule = [&]() {
        for(I i = 0; i < ntables; i++)
        {
            auto const player1 = vec[map[2 * i]];
            auto const player2 = vec[map[2 * i + 1]];
            auto const iplayer1 = std::min(player1, player2);
            auto const iplayer2 = std::max(player1, player2);

            result.push_back(iplayer1);
            result.push_back(iplayer2);
        };
    };

    for(I i = 0; i < m; i++)
    {
        vec[i] = i;
    };

    // top row
    for(I i = 0; i < m / 2; i++)
    {
        map[2 * i] = i;
    };
    map[(m - 1)] = (m - 1);

    auto ival = (m - 1) - 1;
    for(I j = 0; j < ntables - 1; j++)
    {
        auto const ip = 1 + 2 * j;
        map[ip] = ival;
        ival--;
    }

    for(I round = 0; round < numRounds; round++)
    {
        gen_schedule();

        rotate();
    };
}

// ---------------------------------------------------------------------
// CPU code to check the tournament schedule
// For example,
// Each player should play against all other (n-1) players
// No player should play against herself
// Each player should play against exactly only 1 opponent in each round
// ---------------------------------------------------------------------
template <typename I>
static bool check_schedule(I const nplayers, std::vector<I>& result)
{
    if(nplayers <= 1)
    {
        return (false);
    };

    auto const m = nplayers + (nplayers % 2);

    I const numRounds = (m - 1);
    bool const isvalid = (result.size() == (m * numRounds));
    if(!isvalid)
    {
        return (false);
    }

    // check result

    int const nrowA = m;
    int const ncolA = m;

    std::vector<int> A(nrowA * ncolA, -1);
    std::vector<I> is_seen(numRounds, 0);

    // ----------------------------------------------
    // Generate table A[i,j] where
    // player i play against player j at round "A[i,j]"
    // ----------------------------------------------

    I const ntables = m / 2;
    for(I round = 0; round < numRounds; round++)
    {
        for(I i = 0; i < ntables; i++)
        {
            auto const ii = result[2 * i + round * m];
            auto const jj = result[2 * i + 1 + round * m];
            auto const ij = ii + jj * int(m);
            auto const ji = jj + ii * int(m);

            // ---------------------------------------
            // each entry should be assigned only once
            // ---------------------------------------
            bool const isok = (A[ij] == -1) && (A[ji] == -1);
            if(!isok)
            {
                return (false);
            };

            A[ij] = round;
            A[ji] = round;
        };
    }

    // --------------------------
    // check each diagonal should be -1
    // --------------------------
    for(I irow = 0; irow < m; irow++)
    {
        auto const jcol = irow;
        auto const ip = irow + jcol * int(m);
        bool const isok_diag = (A[ip] == -1);
        if(!isok_diag)
        {
            return (false);
        };
    };

    // ---------------------------------------------
    // check each player has played with all other players
    // ---------------------------------------------
    for(I iplayer = 0; iplayer < m; iplayer++)
    {
        // -------------------
        // Array is_seen[iround] counts how many times
        // player j has been seen
        // -------------------
        for(I j = 0; j < numRounds; j++)
        {
            is_seen[j] = 0;
        };

        for(I j = 0; j < m; j++)
        {
            if(j == iplayer)
            {
                continue;
            };

            // ---------
            // check row
            // ---------
            auto const ip = iplayer + j * int(m);
            auto const iround = A[ip];

            bool const is_valid = (0 <= iround) && (iround < numRounds);
            if(!is_valid)
            {
                return (false);
            }

            is_seen[iround]++;
        }

        // ---------------------------------------------------
        // check each player plays  with exactly only 1
        // other player in each round
        // ---------------------------------------------------
        for(auto j = 0; j < numRounds; j++)
        {
            bool const isok = (is_seen[j] == 1);
            if(!isok)
            {
                return (false);
            };
        }
    }

    // ---------------------------------------------
    // each player has played with all other players
    // ---------------------------------------------
    for(auto iplayer = 0; iplayer < m; iplayer++)
    {
        // -------------------
        // zero out is_seen(:)
        // -------------------
        for(I j = 0; j < numRounds; j++)
        {
            is_seen[j] = 0;
        };

        // ------------
        // check column
        // ------------
        for(auto j = 0; j < m; j++)
        {
            if(j == iplayer)
            {
                continue;
            };

            auto const ip = j + iplayer * int(m);
            auto const iround = A[ip];

            bool const is_valid = (0 <= iround) && (iround < numRounds);
            if(!is_valid)
            {
                return (false);
            }

            is_seen[iround]++;
        }

        // ---------------------------------------------------
        // play with exactly only 1 other player in each round
        // ---------------------------------------------------
        for(auto j = 0; j < numRounds; j++)
        {
            bool const isok = (is_seen[j] == 1);
            if(!isok)
            {
                return (false);
            };
        }
    }

    bool const is_valid_schedule = true;
    return (is_valid_schedule);
}

/************** Kernels and device functions for small size*******************/
/*****************************************************************************/

#ifndef RSYEVJ_BDIM
#define RSYEVJ_BDIM 1024 // Max number of threads per thread-block used in rsyevj_small kernel
#endif

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

template <typename T, typename I>
static I get_nb(I const n, bool const need_V)
{
    // ------------------------------------------------------------
    // adjust block size nb so that there are even number of blocks
    //
    // For example, if n = 97, initial nb = 16, then
    // 97/16  = 6.06, so ceil(97,16) = 7
    // we want even number of  blocks, so try nb = 15
    // 97/15 = 6.47
    // 97/14 = 6.93
    // 97/13 = 7.48 ---> so 8 blocks if nb = 13
    // ------------------------------------------------------------

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto is_odd = [](auto n) { return ((n % 2) != 0); };

    I const nb0 = RSYEVJ_BLOCKED_SWITCH(T, need_V) / 2;
    I nb = nb0;
    while(is_odd(ceil(n, nb)))
    {
        nb--;
    };

    return (nb);
}

template <typename I>
static __device__ I idx2D(I const i, I const j, I const ld)
{
    assert((0 <= i) && (i < ld) && (0 <= j));
    return (i + j * static_cast<int64_t>(ld));
};

template <typename T, typename I>
static __device__ void check_symmetry_body(I const n,
                                           T* A_,
                                           I const lda,
                                           I const i_start,
                                           I const i_inc,
                                           I const j_start,
                                           I const j_inc,
                                           int* is_symmetric)
{
    auto A = [=](auto i, auto j) -> T& {
        assert((0 <= i) && (i < n) && (i < lda) && (0 <= j) && (j < n));

        return (A_[idx2D(i, j, lda)]);
    };

    // ----------------------------------------
    // ** note ** atomicAnd works on type "int"
    // not on type "bool"
    // ----------------------------------------

    bool const is_root = (i_start == 0) && (j_start == 0);
    if(is_root)
    {
        *is_symmetric = true;
    };
    __syncthreads();

    for(auto j = j_start; j < n; j += j_inc)
    {
        for(auto i = i_start; i < n; i += i_inc)
        {
            auto const aij = A(i, j);
            auto const aji = A(j, i);
            bool const is_same = (aij == conj(aji));
            if(!is_same)
            {
                // -----------------------
                // matrix is not symmetric
                // -----------------------
                atomicAnd(is_symmetric, false);
                break;
            }
        }
    }
    __syncthreads();
}

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
    assert(use_upper || use_lower);

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto A = [=](auto i, auto j) -> T& {
        assert((0 <= i) && (i < n) && (i < lda) && (0 <= j) && (j < n));

        return (A_[idx2D(i, j, lda)]);
    };

    for(auto j = j_start; j < n; j += j_inc)
    {
        for(auto i = i_start; i < n; i += i_inc)
        {
            bool const is_strictly_lower = (i > j);
            bool const is_strictly_upper = (i < j);
            bool const is_diagonal = (i == j);

            bool const do_assignment
                = (use_upper && is_strictly_upper) || (use_lower && is_strictly_lower);
            if(do_assignment)
            {
                A(j, i) = conj(A(i, j));
            }

            if(is_diagonal)
            {
                A(i, i) = (A(i, i) + conj(A(i, i))) / 2;
            }
        }
    }
    __syncthreads();

#ifdef NDEBUG
#else
    {
        // --------------------------------
        // double check matrix is symmetric
        // --------------------------------
        bool is_symmetric = true;
        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = i_start; i < n; i += i_inc)
            {
                auto const aij = A(i, j);
                auto const aji = A(j, i);
                bool const is_same = (aij == conj(aji));
                if(!is_same)
                {
                    printf(
                        "symmetrize_matrix: use_upper=%d,i=%d,j=%d, abs(aij)=%le, abs(aji)=%le\n",
                        (int)use_upper, (int)i, (int)j, (double)std::abs(aij), (double)std::abs(aji));

                    is_symmetric = false;
                    break;
                }
            }
        }

        assert(is_symmetric);
    }
#endif
}

// ----------------------------------------------------
// make matrix n by n matrix A to be  symmetric or hermitian
//
// launch configuration as dim3(1,1,nbz), dim3(nx,ny,1)
// ----------------------------------------------------
template <typename T, typename I, typename AA, typename Istride>
__global__ static void symmetrize_matrix_kernel(char const uplo,
                                                I const n,
                                                AA A,
                                                I const shiftA,
                                                I const lda,
                                                Istride const strideA,
                                                I const batch_count)
{
    auto const bid_start = hipBlockIdx_z;
    auto const bid_inc = hipGridDim_z;

    auto const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    auto const i_inc = hipBlockDim_x * hipGridDim_x;

    auto const j_start = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;
    auto const j_inc = hipBlockDim_y * hipGridDim_y;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const Ap = load_ptr_batch(A, bid, shiftA, strideA);

        symmetrize_matrix_body(uplo, n, Ap, lda, i_start, i_inc, j_start, j_inc);
    }
}

// ----------------------------------------
// copy m by n submatrix from matrix A to matrix B
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

    auto A = [=](auto i, auto j) -> const T& {
        assert((0 <= i) && (i < m) && (i < lda) && (0 <= j) && (j < n));

        return (A_[idx2D(i, j, lda)]);
    };

    auto B = [=](auto i, auto j) -> T& {
        assert((0 <= i) && (i < m) && (i < ldb) && (0 <= j) && (j < n));

        return (B_[idx2D(i, j, ldb)]);
    };

    for(auto j = j_start; j < n; j += j_inc)
    {
        for(auto i = i_start; i < m; i += i_inc)
        {
            bool const is_upper = (i <= j);
            bool const is_lower = (i >= j);

            bool const do_assignment = use_full || (use_upper && is_upper) || (use_lower && is_lower);
            if(do_assignment)
            {
                B(i, j) = A(i, j);
            }
        }
    }
    __syncthreads();
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
                                  T const alpha_offdiag,
                                  T const beta_diag,
                                  T* A_,
                                  I const lda,
                                  I const i_start,
                                  I const i_inc,
                                  I const j_start,
                                  I const j_inc)
{
    // ------------------------
    // set offdiagonal to alpha_offdiag
    // set diagonal to beta_diag
    // ------------------------

    bool const use_upper = (uplo == 'U') || (uplo == 'u');
    bool const use_lower = (uplo == 'L') || (uplo == 'l');
    bool const use_offdiag = (uplo == 'F') || (uplo == 'f');
    bool const use_full = (!use_upper) && (!use_lower);

    auto A = [=](auto i, auto j) -> T& {
        assert((0 <= i) && (i < m) && (i < lda) && (0 <= j) && (j < n));
        return (A_[idx2D(i, j, lda)]);
    };

    for(I j = j_start; j < n; j += j_inc)
    {
        for(I i = i_start; i < m; i += i_inc)
        {
            bool const is_diag = (i == j);
            bool const is_strictly_upper = (i < j);
            bool const is_strictly_lower = (i > j);

            bool const do_assignment
                = (use_full || (use_offdiag && (!is_diag)) || (use_lower && is_strictly_lower)
                   || (use_upper && is_strictly_upper));

            if(do_assignment)
            {
                A(i, j) = alpha_offdiag;
            }

            if(is_diag && (!use_offdiag))
            {
                A(i, i) = beta_diag;
            }
        }
    }
    __syncthreads();
}

// --------------------------------------------
// can be used to set matrix to identity matrix
//
// launch as dim3(nbx,nby,nbz), dim3(nx,ny,1)
// --------------------------------------------
template <typename T, typename I, typename UA, typename Istride>
__global__ static void laset_kernel(char const c_uplo,
                                    I const m,
                                    I const n,
                                    T const alpha_offdiag,
                                    T const beta_diag,
                                    UA A,
                                    I const shiftA,
                                    I const lda,
                                    Istride const strideA,
                                    I const batch_count)
{
    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    I const j_start = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;
    I const j_inc = hipBlockDim_y * hipGridDim_y;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const A_p = load_ptr_batch(A, bid, shiftA, strideA);

        laset_body(m, n, alpha_offdiag, beta_diag, A_p, lda, i_start, i_inc, j_start, j_inc);
    }
}

// --------------------------------------------------
// symmetrically reorder matrix so that the set of independent pairs
// becomes (0,1), (2,3), ...
//
// assume launch as  dim3( nbx, nby, batch_count), dim3(nx,ny,1)
// ** NOTE ** assume row_map[] and col_map[] map the last block
// also to the last block
// this means
// row_map[ (nblocks-1) ] == (nblocks-1)
// col_map[ (nblocks-1) ] == (nblocks-1)
// -------------------------------------------------------------
template <typename T, typename I, typename Istride, typename UA, typename UC>
__global__ static void reorder_kernel(char c_direction,
                                      I const n,
                                      I const nb,
                                      I const* const row_map,
                                      I const* const col_map,
                                      UA AA,
                                      I const shiftA,
                                      I const ldA,
                                      Istride strideA,
                                      UC CC,
                                      I const shiftC,
                                      I const ldC,
                                      Istride const strideC,
                                      I const batch_count)
{
    bool const is_forward = (c_direction == 'F') || (c_direction == 'f');
    // ----------------------------------
    // use identity map if map == nullptr
    // ----------------------------------
    bool const has_row_map = (row_map != nullptr);
    bool const has_col_map = (col_map != nullptr);

    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const ib_start = hipBlockIdx_x;
    I const ib_inc = hipGridDim_x;

    I const jb_start = hipBlockIdx_y;
    I const jb_inc = hipGridDim_y;

    // -----------------------
    // indexing within a block
    // -----------------------
    I const i_start = hipThreadIdx_x;
    I const i_inc = hipBlockDim_x;

    I const j_start = hipThreadIdx_y;
    I const j_inc = hipBlockDim_y;

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto const nblocks = ceil(n, nb);

    // -----------------------------------------
    // map the last block also to the last block
    // -----------------------------------------
    if(has_row_map)
    {
        assert(row_map[(nblocks - 1)] == (nblocks - 1));
    }
    if(has_col_map)
    {
        assert(col_map[(nblocks - 1)] == (nblocks - 1));
    }

    // ----------------------
    // size of the i-th block
    // ----------------------
    auto bsize = [=](auto iblock) {
        auto const iremain = n - (nblocks - 1) * nb;
        return ((iblock == (nblocks - 1)) ? iremain : nb);
    };

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const dA = load_ptr_batch<T>(AA, bid, shiftA, strideA);
        T* const dC = load_ptr_batch<T>(CC, bid, shiftC, strideC);

        for(auto jblock = jb_start; jblock < nblocks; jblock += jb_inc)
        {
            for(auto iblock = ib_start; iblock < nblocks; iblock += ib_inc)
            {
                auto const iblock_old = (has_row_map) ? row_map[iblock] : iblock;
                auto const jblock_old = (has_col_map) ? col_map[jblock] : jblock;

                auto const ii = iblock * nb;
                auto const jj = jblock * nb;
                auto const ii_old = iblock_old * nb;
                auto const jj_old = jblock_old * nb;

                // -----------------------------
                // pointer to start of
                // (iblock_old, jblock_old) in A
                // (iblock,jblock) in C
                // -----------------------------
                Istride const offset_A
                    = (is_forward) ? idx2D(ii_old, jj_old, ldA) : idx2D(ii, jj, ldA);

                Istride const offset_C
                    = (is_forward) ? idx2D(ii, jj, ldC) : idx2D(ii_old, jj_old, ldC);
                T const* const A = dA + offset_A;
                T* const C = dC + offset_C;

                {
                    char const c_uplo = 'A';
                    I const mm = bsize(iblock);
                    I const nn = bsize(jblock);

                    assert(bsize(iblock_old) == bsize(iblock));
                    assert(bsize(jblock_old) == bsize(jblock));

                    lacpy_body(c_uplo, mm, nn, A, ldA, C, ldC, i_start, i_inc, j_start, j_inc);
                }
            }
        }
    }
}

// ---------------------
// copy diagonal entries
//
// launch as dim3(nbx,1,nbz), dim3(nx,1,1)
// ---------------------
template <typename T, typename I, typename AA, typename Istride>
__global__ static void copy_diagonal_kernel(I const n,
                                            AA A,
                                            I const shiftA,
                                            I const lda,
                                            Istride const strideA,

                                            T* const W,
                                            Istride const strideW,
                                            I const batch_count)
{
    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    I const bid_start = hipBlockIdx_z;

    I const bid_inc = hipGridDim_z;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const A_p = load_ptr_batch(A, bid, shiftA, strideA);
        T* const W_p = W + bid * strideW;

        for(auto i = i_start; i < n; i += i_inc)
        {
            T const aii = A_p[idx2D(i, i, lda)];
            W_p[i] = aii;
        };
    }
}

// ---------------------------------
// copy m by n submatrix from A to C
// launch as
// dim3(nbx,nby,min(max_blocks,batch_count)), dim3(nx,ny,1)
// ---------------------------------
template <typename T, typename I, typename AA, typename CC, typename Istride>
__global__ static void lacpy_kernel(char const uplo,
                                    I const m,
                                    I const n,
                                    AA A,
                                    I const shiftA,
                                    I const lda,
                                    Istride strideA,
                                    CC C,
                                    I const shiftC,
                                    I const ldc,
                                    Istride strideC,
                                    I const batch_count)
{
    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_inc = hipGridDim_x * hipBlockDim_x;
    I const j_inc = hipGridDim_y * hipBlockDim_y;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const j_start = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const Ap = load_ptr_batch(A, bid, shiftA, strideA);
        T* const Cp = load_ptr_batch(C, bid, shiftC, strideC);

        lacpy_body(uplo, m, n, Ap, lda, Cp, ldc, i_start, i_inc, j_start, j_inc);
    }
}

template <typename T, typename I, typename AA, typename CC, typename Istride>
static void lacpy(rocblas_handle handle,
                  char const c_uplo,
                  I const m,
                  I const n,
                  AA A,
                  I const shiftA,
                  I const lda,
                  Istride strideA,
                  CC C,
                  I const shiftC,
                  I const ldc,
                  Istride const strideC,
                  I const batch_count)
{
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    I const max_blocks = 64 * 1000;
    auto const nx = 32;
    auto const ny = RSYEVJ_BDIM / nx;

    auto const nbx = std::min(max_blocks, ceil(m, nx));
    auto const nby = std::min(max_blocks, ceil(n, ny));
    auto const nbz = std::min(max_blocks, batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    ROCBLAS_LAUNCH_KERNEL((lacpy_kernel<T, I, AA, CC, Istride>), dim3(nbx, nby, nbz),
                          dim3(nx, ny, 1), 0, stream, c_uplo, m, n, A, shiftA, lda, strideA, C,
                          shiftC, ldc, strideC, batch_count);
}

// --------------------------------------
// index calculations for compact storage
// --------------------------------------
template <typename I>
static __device__ I idx_upper(I const i, I const j, I const n)
{
    assert((0 <= i) && (i < n) && (0 <= j) && (j < n));
    return (i + (j * (j + 1)) / 2);
};

template <typename I>
static __device__ I idx_lower(I const i, I const j, I const n)
{
    assert((0 <= i) && (i < n) && (0 <= j) && (j < n));
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
        rt2 = (dble(acmx) / dble(rt1)) * dble(acmn) - (dble(b) / dble(rt1)) * dble(b);
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
#ifdef NDEBUG
#else
    // ---------------------------------------------------------
    // double check results
    //
    // [ cs1  sn1 ]  [ a       b]  [ cs1   -sn1 ] = [ rt1    0  ]
    // [-sn1  cs1 ]  [ b       c]  [ sn1    cs1 ]   [ 0      rt2]
    // ---------------------------------------------------------

    // -----------------------------------------
    // [ cs1  sn1 ]  [ a       b]  -> [a11  a12]
    // [-sn1  cs1 ]  [ b       c]     [a21  a22]
    // -----------------------------------------
    auto const a11 = cs1 * a + sn1 * b;
    auto const a12 = cs1 * b + sn1 * c;
    auto const a21 = (-sn1) * a + cs1 * b;
    auto const a22 = (-sn1) * b + cs1 * c;

    // -----------------------------------------
    // [a11 a12]  [ cs1   -sn1 ] = [ rt1    0  ]
    // [a21 a22]  [ sn1    cs1 ]   [ 0      rt2]
    // -----------------------------------------

    auto e11 = a11 * cs1 + a12 * sn1 - rt1;
    auto e12 = a11 * (-sn1) + a12 * cs1;
    auto e21 = a21 * cs1 + a22 * sn1;
    auto e22 = a21 * (-sn1) + a22 * cs1 - rt2;

    auto const anorm = std::sqrt(std::norm(rt1) + std::norm(rt2));

    auto const enorm = std::sqrt(std::norm(e11) + std::norm(e12) + std::norm(e21) + std::norm(e22));

    auto const tol = 1e-6;
    bool const isok = (enorm <= tol * anorm);
    if(!isok)
    {
        printf("dlaev2: enorm = %le, anorm= %le\n", (double)enorm, (double)anorm);
        printf("a = %le, b = %le, c = %le\n", (double)a, (double)b, (double)c);
    };
    assert(isok);
#endif
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

#ifdef NDEBUG
#else
    // --------------------
    // double check results
    // --------------------
    // [cs1  conj(sn1) ]  [ a        b]  [ cs1   -conj(sn1) ] = [ rt1    0  ]
    // [-sn1  cs1      ]  [ conj(b)  c]  [ sn1    cs1       ]   [ 0      rt2]

    // -------------------------------------------------
    // [cs1  conj(sn1) ]  [ a        b]  -> [a11   a12]
    // [-sn1  cs1      ]  [ conj(b)  c]     [a21   a22]
    // -------------------------------------------------
    auto const a11 = cs1 * a + conj(sn1) * b;
    auto const a12 = cs1 * b + conj(sn1) * c;
    auto const a21 = (-sn1) * a + cs1 * conj(b);
    auto const a22 = (-sn1) * b + cs1 * c;

    // -----------------------------------------------
    // [a11 a12]  [ cs1   -conj(sn1) ] = [ rt1    0  ]
    // [a21 a22]  [ sn1    cs1       ]   [ 0      rt2]
    // -----------------------------------------------

    auto const anorm = std::sqrt(std::norm(rt1) + std::norm(rt2));

    auto const e11 = a11 * cs1 + a12 * sn1 - rt1;
    auto const e12 = a11 * (-conj(sn1)) + a12 * cs1;
    auto const e21 = a21 * cs1 + a22 * sn1;
    auto const e22 = a21 * (-conj(sn1)) + a22 * cs1 - rt2;

    auto const enorm = std::sqrt(std::norm(e11) + std::norm(e12) + std::norm(e21) + std::norm(e22));

    auto const tol = 1e-6;
    auto isok = (enorm <= tol * anorm);
    if(!isok)
    {
        printf("zlaev2: enorm=%le, anorm=%le\n", (double)enorm, (double)anorm);
        printf("a = (%le,%le), b = (%le, %le), c = (%le, %le)\n", std::real(a), std::imag(a),
               std::real(b), std::imag(b), std::real(c), std::imag(c));
    }
    assert(isok);
#endif

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
// calculate the Frobenius-norm of n by n (complex) matrix
//
// ** NOTE **  answer is returned in dwork[0]
// -------------------------------------
template <typename T, typename I, typename S>
static __device__ void cal_norm_body(I const m,
                                     I const n,
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
    bool const is_root = (i_start == 0) && (j_start == 0);

    S const zero = 0.0;
    constexpr bool use_serial = false;

    if(use_serial)
    {
        // ------------------------------
        // simple serial code should work
        // ------------------------------

        if(is_root)
        {
            S dsum = 0.0;
            for(auto j = 0; j < n; j++)
            {
                for(auto i = 0; i < m; i++)
                {
                    bool const is_diag = (i == j);
                    auto const aij = (is_diag && (!include_diagonal)) ? zero : A(i, j);

                    dsum += std::norm(aij);
                }
            }
            dwork[0] = std::sqrt(dsum);
        }

        __syncthreads();
        return;
    }

    // ------------------
    // initialize dwork(:)
    // ------------------
    if(is_root)
    {
        dwork[0] = zero;
    }

    __syncthreads();

    {
        S dsum = zero;
        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = i_start; i < m; i += i_inc)
            {
                bool const is_diag = (i == j);
                auto const aij = (is_diag && (!include_diagonal)) ? zero : A(i, j);
                dsum += std::norm(aij);
            }
        }

        if(dsum != 0)
        {
            atomicAdd(&(dwork[0]), dsum);
        }
    }

    __syncthreads();

    if(is_root)
    {
        dwork[0] = std::sqrt(dwork[0]);
    }
    __syncthreads();

    return;
}

/** kernel to setup pointer arrays in preparation
 * for calls to batched GEMM and for copying data
 *
 * launch as dim3(1,1,batch_count), dim3(32,1,1)
**/
template <typename T, typename I, typename Istride, typename AA, typename BB, typename CC>
__global__ static void setup_ptr_arrays_kernel(

    I const n,
    I const nb,

    AA A,
    I const shiftA,
    I const lda,
    Istride const strideA,

    BB Atmp,
    I const shiftAtmp,
    I const ldatmp,
    Istride const strideAtmp,

    CC Vtmp,
    I const shiftVtmp,
    I const ldvtmp,
    Istride const strideVtmp,

    T* const Aj,
    T* const Vj,

    I* const completed,

    T* Vj_ptr_array[],
    T* Aj_ptr_array[],
    T* Vj_last_ptr_array[],
    T* Aj_last_ptr_array[],

    T* A_row_ptr_array[],
    T* A_col_ptr_array[],
    T* A_row_last_ptr_array[],
    T* A_col_last_ptr_array[],
    T* A_ptr_array[],

    T* Atmp_row_ptr_array[],
    T* Atmp_col_ptr_array[],
    T* Atmp_row_last_ptr_array[],
    T* Atmp_col_last_ptr_array[],
    T* Atmp_ptr_array[],

    T* Vtmp_row_ptr_array[],
    T* Vtmp_col_ptr_array[],
    T* Vtmp_row_last_ptr_array[],
    T* Vtmp_col_last_ptr_array[],
    T* Vtmp_ptr_array[],

    T* A_diag_ptr_array[],
    T* A_last_diag_ptr_array[],

    I const batch_count)
{
    auto const bid_start = hipBlockIdx_z;
    auto const bid_inc = hipGridDim_z;

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto const nblocks = ceil(n, nb);
    auto const nblocks_even = nblocks + (nblocks % 2);
    auto const nblocks_half = nblocks_even / 2;

    auto const idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        bool const is_completed = completed[bid + 1];
        if(is_completed)
        {
            continue;
        };

        T* const A_p = load_ptr_batch(A, bid, shiftA, strideA);
        T* const Atmp_p = load_ptr_batch(Atmp, bid, shiftAtmp, strideAtmp);
        T* const Vtmp_p = load_ptr_batch(Vtmp, bid, shiftVtmp, strideVtmp);

        auto const ibatch = atomicAdd(&(completed[0]), 1);

        A_ptr_array[ibatch] = A_p;
        Atmp_ptr_array[ibatch] = Atmp_p;
        Vtmp_ptr_array[ibatch] = Vtmp_p;

        auto const irow_last = (nblocks_half - 1) * (2 * nb);
        auto const jcol_last = (nblocks_half - 1) * (2 * nb);

        A_row_last_ptr_array[ibatch] = A_p + shiftA + idx2D(irow_last, 0, lda);
        A_col_last_ptr_array[ibatch] = A_p + shiftA + idx2D(0, jcol_last, lda);

        A_last_diag_ptr_array[ibatch] = A_p + shiftA + idx2D(irow_last, jcol_last, lda);

        Atmp_row_last_ptr_array[ibatch] = Atmp_p + shiftAtmp + idx2D(irow_last, 0, ldatmp);
        Atmp_col_last_ptr_array[ibatch] = Atmp_p + shiftAtmp + idx2D(0, jcol_last, ldatmp);

        Vtmp_row_last_ptr_array[ibatch] = Vtmp_p + shiftVtmp + idx2D(irow_last, 0, ldvtmp);
        Vtmp_col_last_ptr_array[ibatch] = Vtmp_p + shiftVtmp + idx2D(0, jcol_last, ldvtmp);

        {
            Istride const strideVj = (nblocks_half * (2 * nb) * (2 * nb));

            Vj_ptr_array[ibatch] = Vj + ibatch * strideVj;
            Vj_last_ptr_array[ibatch]
                = Vj + ibatch * strideVj + (nblocks_half - 1) * ((2 * nb) * (2 * nb));
        }

        {
            Istride const strideAj = (nblocks_half * (2 * nb) * (2 * nb));

            Aj_ptr_array[ibatch] = Aj + ibatch * strideAj;
            Aj_last_ptr_array[ibatch]
                = Aj + ibatch * strideAj + (nblocks_half - 1) * ((2 * nb) * (2 * nb));
        }

        for(auto i = i_start; i < (nblocks_half - 1); i += i_inc)
        {
            auto const ip = i + ibatch * (nblocks_half - 1);

            {
                auto const irow = i * (2 * nb);
                auto const jcol = 0;
                A_row_ptr_array[ip] = A_p + shiftA + idx2D(irow, jcol, lda);
                Atmp_row_ptr_array[ip] = Atmp_p + shiftAtmp + idx2D(irow, jcol, ldatmp);
                Vtmp_row_ptr_array[ip] = Vtmp_p + shiftVtmp + idx2D(irow, jcol, ldvtmp);
            }

            {
                auto const irow = 0;
                auto const jcol = i * (2 * nb);

                A_col_ptr_array[ip] = A_p + shiftA + idx2D(irow, jcol, lda);
                Atmp_col_ptr_array[ip] = Atmp_p + shiftAtmp + idx2D(irow, jcol, ldatmp);
                Vtmp_col_ptr_array[ip] = Vtmp_p + shiftVtmp + idx2D(irow, jcol, ldvtmp);

                {
                    I const irow = (2 * nb) * i;
                    I const jcol = irow;
                    A_diag_ptr_array[ip] = A_p + shiftA + idx2D(irow, jcol, lda);
                }
            }
        }

    } // end for bid
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
    constexpr int idebug = 1;
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

    I const ldv = n;

    I const i_start = tix;
    I const i_inc = dimx;
    I const j_start = tiy;
    I const j_inc = dimy;

    // ---------------------------------------
    // reuse storage
    // ---------------------------------------
    S* const dwork = cosine;

    auto const num_rounds = (n_even - 1);
    auto const ntables = (n_even / 2);

    // ----------------------------------------------------
    // schedule_(:)  is array of size n_even * (n_even - 1)
    // that contains the tournament schedule for n_even number
    // of players
    // ----------------------------------------------------
    auto schedule = [=](auto i1, auto itable, auto iround) {
        assert((0 <= i1) && (i1 < 2) && (0 <= itable) && (itable < ntables) && (0 <= iround)
               && (iround < num_rounds));

        return (schedule_[i1 + itable * 2 + iround * n_even]);
    };

    auto V = [=](auto i, auto j) -> T& {
        assert((0 <= i) && (i < n) && (0 <= j) && (j < n) && (i < ldv));

        return (V_[i + j * ldv]);
    };
    auto A = [=](auto i, auto j) -> T& {
        assert((0 <= i) && (i < n) && (0 <= j) && (j < n) && (i < lda));

        return (A_[i + j * lda]);
    };

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
        }

        {
            // ----------------
            // symmetrize matrix
            // ----------------

            char const c_uplo = (is_upper) ? 'U' : 'L';
            I const nn = n;
            I const ld1 = (is_same) ? lda : nn;

            symmetrize_matrix_body(c_uplo, nn, A_, ld1, i_start, i_inc, j_start, j_inc);

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

        laset_body(c_uplo, mm, nn, alpha_offdiag, beta_diag, V_, ld1, i_start, i_inc, j_start, j_inc);

        __syncthreads();
    }

    double norm_A = 0;
    {
        bool const need_diagonal = true;
        I ld1 = n;
        S* Swork = (S*)dwork;
        auto const mm = n;
        auto const nn = n;
        cal_norm_body(mm, nn, A_, ld1, Swork, i_start, i_inc, j_start, j_inc, need_diagonal);
        norm_A = Swork[0];
    }

    bool has_converged = false;

    // ----------------------------------------------------------
    // NOTE: need to preserve value of isweep outside of for loop
    // ----------------------------------------------------------
    I isweep = 0;

#ifdef NDEBUG
#else
    {
        // --------------------------------
        // extra check that A_ is symmetric
        // --------------------------------
        auto ld1 = (A_ == dA_) ? lda : n;
        int is_symmetric = true;
        check_symmetry_body(n, A_, ld1, i_start, i_inc, j_start, j_inc, &is_symmetric);
        assert(is_symmetric);
    }
#endif

    S norm_offdiag = 0;
    for(isweep = 0; isweep < max_sweeps; isweep++)
    {
        // ----------------------------------------------------------
        // check convergence by computing norm of off-diagonal matrix
        // ----------------------------------------------------------
        __syncthreads();

        bool const need_diagonal = false;
        S* const Swork = (S*)dwork;

        auto const mm = n;
        auto const nn = n;
        cal_norm_body(mm, nn, A_, lda, Swork, i_start, i_inc, j_start, j_inc, need_diagonal);
        norm_offdiag = Swork[0];

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
                bool const is_valid_p = (0 <= p) && (p < n);
                bool const is_valid_q = (0 <= q) && (q < n);
                bool const is_valid_pq = is_valid_p && is_valid_q;

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
                S rt1 = 0;
                S rt2 = 0;
                laev2<T, S>(App, Apq, Aqq, rt1, rt2, cs1, sn1);

                //  ----------------------------------
                //  We have
                //
                //  J' * [App  Apq] * J = [rt1   0  ]
                //       [Apq' Aqq]       [0     rt2]
                //
                //
                //  J = [cs1   -conj(sn1)]
                //      [sn1    cs1      ]
                //  ----------------------------------

                auto const J11 = cs1;
                auto const J12 = -conj(sn1);
                auto const J21 = sn1;
                auto const J22 = cs1;

                // ------------------------
                // J' is conj(transpose(J))
                // ------------------------
                auto const Jt11 = conj(J11);
                auto const Jt12 = conj(J21);
                auto const Jt21 = conj(J12);
                auto const Jt22 = conj(J22);

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

                    // ---------------------
                    // [Jt11, Jt12 ] * [Api]
                    // [Jt21, Jt22 ]   [Aqi]
                    // ---------------------

                    A(p, i) = Jt11 * Api + Jt12 * Aqi;
                    A(q, i) = Jt21 * Api + Jt22 * Aqi;

                } // end for i

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
                bool const is_valid_p = (0 <= p) && (p < n);
                bool const is_valid_q = (0 <= q) && (q < n);
                bool const is_valid_pq = is_valid_p && is_valid_q;
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

                auto const J11 = cs1;
                auto const J12 = -conj(sn1);
                auto const J21 = sn1;
                auto const J22 = cs1;

                // -------------------------------
                // update columns p, q in matrix A
                // -------------------------------
                for(auto i = i_start; i < n; i += i_inc)
                {
                    {
                        auto const Aip = A(i, p);
                        auto const Aiq = A(i, q);

                        // -----------------------
                        // [Aip, Aiq] * [J11, J12]
                        //              [J21, J22]
                        // -----------------------

                        A(i, p) = Aip * J11 + Aiq * J21;
                        A(i, q) = Aip * J12 + Aiq * J22;
                    }

                    // --------------------
                    // update eigen vectors
                    // --------------------
                    if(need_V)
                    {
                        auto const Vip = V(i, p);
                        auto const Viq = V(i, q);

                        // -----------------------
                        // [Vip, Viq] * [J11, J12]
                        //              [J21, J22]
                        // -----------------------

                        V(i, p) = Vip * J11 + Viq * J21;
                        V(i, q) = Vip * J12 + Viq * J22;
                    }
                } // end for i

#ifdef NDEBUG
#else
                if(i_start == 0)
                {
                    // ------------------------------
                    // double check A(p,q) and A(q,p)
                    // ------------------------------
                    auto const tol = 1e-6;
                    bool const isok_pq = std::abs(A(p, q)) <= tol * norm_A;
                    bool const isok_qp = std::abs(A(q, p)) <= tol * norm_A;
                    bool const isok = isok_pq && isok_qp;
                    if(!isok)
                    {
                        printf("n=%d,need_V=%d,j=%d,iround=%d,isweep=%d\n", (int)n, (int)need_V,
                               (int)j, (int)iround, (int)isweep);
                        printf("p=%d,q=%d,abs(A(p,q))=%le,abs(A(q,p))=%le\n", (int)p, (int)q,
                               (double)std::abs(A(p, q)), (double)std::abs(A(q, p)));
                    }
                    assert(isok);
                }

#endif

                // -------------------------------------
                // explicitly set A(p,q), A(q,p) be zero
                // otherwise, abs(A(p,q)) may still be around
                // machine epsilon
                // -------------------------------------
                {
                    if(i_start == 0)
                    {
                        A(p, q) = 0;
                        A(q, p) = 0;
                    }
                }

            } // end for j
            __syncthreads();
        } // end for iround

    } // for isweeps

    __syncthreads();

    // -----------------
    // copy eigen values
    // -----------------
    if(j_start == 0)
    {
        for(auto i = i_start; i < n; i += i_inc)
        {
            W[i] = std::real(A(i, i));
        }
    }
    __syncthreads();

    if(idebug >= 2)
    {
        if((i_start == 0) && (j_start == 0))
        {
            for(auto i = 0; i < n; i++)
            {
                printf("W(%d) = %le;\n", (int)i + 1, (double)W[i]);
            }
        }
    }

    *info = (has_converged) ? 0 : 1;
    *n_sweeps = (has_converged) ? isweep : max_sweeps;
    *residual = norm_offdiag;

    // debug
    if((i_start == 0) && (j_start == 0))
    {
        printf("isweep=%d, abstol=%le,norm_offdiag = %le, norm_A = %le\n", (int)isweep,
               (double)abstol, (double)norm_offdiag, (double)norm_A);
    }

    // -----------------
    // sort eigen values
    // -----------------
    if(need_sort)
    {
        I* const map = (need_V) ? (I*)dwork : nullptr;

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

template <typename T, typename I, typename S, typename Istride, typename AA>
__global__ static void cal_Gmat_kernel(I const n,
                                       I const nb,
                                       AA A,
                                       I const shiftA,
                                       I const lda,
                                       Istride const strideA,

                                       S* const Gmat_,
                                       bool const include_diagonal_values,

                                       I const batch_count)
{
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto const nblocks = ceil(n, nb);
    auto const nb_last = n - (nblocks - 1) * nb;

    auto const idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto Gmat = [=](auto ib, auto jb, auto bid) -> S& {
        return (Gmat_[ib + jb * nblocks + bid * (nblocks * nblocks)]);
    };

    auto bsize = [=](auto iblock) { return ((iblock == (nblocks - 1)) ? nb_last : nb); };

    auto const bid_start = hipBlockIdx_z;
    auto const bid_inc = hipGridDim_z;

    auto const ib_start = hipBlockIdx_x;
    auto const jb_start = hipBlockIdx_y;

    auto const ib_inc = hipGridDim_x;
    auto const jb_inc = hipGridDim_y;

    auto const i_start = hipThreadIdx_x;
    auto const j_start = hipThreadIdx_y;

    auto const i_inc = hipBlockDim_x;
    auto const j_inc = hipBlockDim_y;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const Ap_ = load_ptr_batch(A, bid, shiftA, strideA);

        auto Ap = [=](auto i, auto j) -> const T& { return (Ap_[idx2D(i, j, lda)]); };

        for(I jb = jb_start; jb < nblocks; jb += jb_inc)
        {
            for(I ib = ib_start; ib < nblocks; ib += ib_inc)
            {
                bool const is_diag_block = (ib == jb);

                bool const need_diagonal
                    = (is_diag_block && include_diagonal_values) || (!is_diag_block);

                auto const ni = bsize(ib);
                auto const nj = bsize(jb);

                auto const ii = ib * nb;
                auto const jj = jb * nb;
                I const ldgmat = nblocks;
                cal_norm_body(ni, nj, &(Gmat(ib, jb, bid)), ldgmat, &(Ap(ii, jj)), i_start, i_inc,
                              j_start, j_inc, need_diagonal);
            }
        }
    }
}

// -----------------------------------------------------
// sum entries in Gmat(nblocks,nblocks, batch_count)
// assume launch as
// <<< dim3(1,1,batch_count), dim3(nx,ny,1),
//      sizeof(S), stream>>>
// -----------------------------------------------------
template <typename S, typename I>
__global__ static void
    sum_Gmat(I const n, I const nb, S* const Gmat_, S* const Gnorm_, I const batch_count)
{
    auto const bid_start = hipBlockIdx_z;
    auto const bid_inc = hipGridDim_z;

    auto const i_start = hipThreadIdx_x;
    auto const i_inc = hipBlockDim_x;

    auto const j_start = hipThreadIdx_y;
    auto const j_inc = hipBlockDim_y;

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };
    auto const nblocks = ceil(n, nb);

    auto Gmat = [=](auto i, auto j, auto bid) -> const S& {
        return (Gmat_[i + j * nblocks + bid * (nblocks * nblocks)]);
    };

    auto Gnorm = [=](auto bid) -> S& { return (Gnorm_[(bid - 1)]); };

    extern __shared__ S sh_mem[];

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        bool const is_root = ((i_start == 0) && (j_start == 0));

        if(is_root)
        {
            sh_mem[0] = 0;
        }

        __syncthreads();

        S dsum = 0;
        for(I j = j_start; j < nblocks; j += j_inc)
        {
            for(I i = i_start; i < nblocks; i += i_inc)
            {
                dsum += std::norm(Gmat(i, j, bid));
            }
        }

        if(dsum != 0)
        {
            atomicAdd(&(sh_mem[0]), dsum);
        };

        __syncthreads();

        if(is_root)
        {
            Gnorm(bid) = std::sqrt(sh_mem[0]);
        };
    }
}

// ------------------------------------------------------
// assume launch as   dim3(nbx,1,1), dim3(nx,1,1)
// ------------------------------------------------------
template <typename S, typename I, typename Istride>
__global__ static void set_completed(I const n,
                                     I const nb,
                                     S* const Anorm,
                                     S const abstol,
                                     S* const Gnorm,

                                     I completed[],
                                     I const batch_count)
{
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto const nblocks = ceil(n, nb);

    auto const bid_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    auto const bid_inc = hipBlockDim_x * hipGridDim_x;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        S const anorm = Anorm[bid];
        S const gnorm = Gnorm[bid];
        bool const is_completed = (gnorm <= abstol * anorm);

        // -----------------
        // note use "bid + 1"
        // -----------------
        completed[bid + 1] = is_completed;

        if(is_completed)
        {
            atomicAdd(&(completed[0]), 1);
        };
    }
}

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
    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
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
        // rsyevj_get_dims(n, RSYEVJ_BDIM, &ddx, &ddy);

        bool const need_V = (esort != rocblas_esort_none);

        // shared memory
        auto const ntables = half_n;
        extern __shared__ double lmem[];
        S* cosines_res = reinterpret_cast<S*>(lmem);
        T* sines_diag = reinterpret_cast<T*>(cosines_res + ntables);
        T* pfree = reinterpret_cast<T*>(sines_diag + ntables);
        T* A_ = pfree;
        pfree += n * n;
        T* V_ = (need_V) ? pfree : nullptr;

        // extra check whether to use A_ and V_ in LDS
        {
            // auto const max_lds = 64 * 1000;
            auto const max_lds = (sizeof(T) + sizeof(S)) * n;

            // ----------------------------------------------------------
            // array cosine also used in comuputing matrix Frobenius norm
            // ----------------------------------------------------------
            size_t const size_cosine = sizeof(S) * n;
            size_t const size_sine = sizeof(T) * ntables;

            size_t const size_A = sizeof(T) * n * n;
            size_t const size_V = (need_V) ? sizeof(T) * n * n : 0;

            size_t const total_size = size_cosine + size_sine + size_A + size_V;

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
            assert((size_cosine + size_sine) <= max_lds);
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

#if(0)
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
#endif

#if(0)
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
#endif

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
                                           size_t* size_dwork)
{
    // set workspace to zero
    {
        *size_Acpy = 0;
        *size_J = 0;
        *size_dwork = 0;
    }

    // quick return
    if(n <= 1 || batch_count == 0)
    {
        return;
    }

    bool const need_V = (evect != rocblas_evect_none);

    if(n <= RSYEVJ_BLOCKED_SWITCH(T, need_V))
    {
        auto const n_even = n + (n % 2);
        *size_dwork = sizeof(I) * (n_even * (n_even - 1));
        return;
    }

    auto const ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };
    auto const is_even = [](auto n) { return ((n % 2) == 0); };

    bool const rsyevj_need_V = true;
    I const nb = get_nb(n, rsyevj_need_V);

    I const n_even = n + (n % 2);
    I const half_n = n_even / 2;

    I const nblocks = ceil(n, nb);
    I const nblocks_even = nblocks + (nblocks % 2);
    I const nblocks_half = nblocks_even / 2;
    assert(is_even(nblocks));

    *size_Acpy = (sizeof(T) * n * n) * batch_count;

    // size of copy of eigen vectors
    if(need_V)
    {
        *size_J = (sizeof(T) * n * n) * batch_count;
    }

    {
        // -----------------------------------------------------
        // other arrays allocated out of a single dwork(:) array
        // -----------------------------------------------------
        size_t total_bytes = 0;

        size_t const size_completed = sizeof(I) * (batch_count + 1);
        total_bytes += size_completed;

        size_t const size_Vj_bytes = sizeof(T) * (nb * 2) * (nb * 2) * (nblocks_half)*batch_count;
        size_t const size_Aj_bytes = sizeof(T) * (nb * 2) * (nb * 2) * (nblocks_half)*batch_count;

        total_bytes += size_Vj_bytes + size_Aj_bytes;

        size_t const size_Vj_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Aj_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;

        total_bytes += size_Vj_ptr_array + size_Aj_ptr_array;

        size_t const size_Vj_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Aj_last_ptr_array = sizeof(T*) * 1 * batch_count;

        total_bytes += size_Vj_last_ptr_array + size_Aj_last_ptr_array;

        size_t const size_A_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_row_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_A_col_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_A_ptr_array = sizeof(T*) * batch_count;

        total_bytes += size_A_row_ptr_array + size_A_col_ptr_array + size_A_row_last_ptr_array
            + size_A_col_last_ptr_array + size_A_ptr_array;

        size_t const size_Atmp_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Atmp_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Atmp_row_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Atmp_col_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Atmp_ptr_array = sizeof(T*) * batch_count;

        total_bytes += size_Atmp_row_ptr_array + size_Atmp_col_ptr_array
            + size_Atmp_row_last_ptr_array + size_Atmp_col_last_ptr_array + size_Atmp_ptr_array;

        size_t const size_Vtmp_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Vtmp_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Vtmp_row_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Vtmp_col_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Vtmp_ptr_array = sizeof(T) * batch_count;

        total_bytes += size_Vtmp_row_ptr_array + size_Vtmp_col_ptr_array
            + size_Vtmp_row_last_ptr_array + size_Vtmp_col_last_ptr_array + size_Vtmp_ptr_array;

        size_t const size_A_diag_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_last_diag_ptr_array = sizeof(T*) * 1 * batch_count;

        total_bytes += size_A_diag_ptr_array + size_A_last_diag_ptr_array;

        size_t const size_Gmat = sizeof(S) * (nblocks * nblocks) * batch_count;
        size_t const size_Gmat_ptr_array = sizeof(S*) * batch_count;
        size_t const size_Gmat_norm = sizeof(S) * batch_count;
        size_t const size_Amat_norm = sizeof(S) * batch_count;

        total_bytes += size_Gmat + size_Gmat_ptr_array;
        total_bytes += size_Gmat_norm + size_Amat_norm;

        size_t const size_schedule_small = sizeof(I) * (2 * nb) * ((2 * nb) - 1);
        size_t const size_schedule_large = sizeof(I) * nblocks_even * (nblocks_even - 1);

        total_bytes += size_schedule_small + size_schedule_large;

        *size_dwork = total_bytes;
    }
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

template <bool BATCHED, bool STRIDED, typename T, typename I, typename S, typename U, typename Istride>
rocblas_status rocsolver_rsyevj_rheevj_template(rocblas_handle handle,
                                                const rocblas_esort esort,
                                                const rocblas_evect evect,
                                                const rocblas_fill uplo,
                                                const I n,
                                                U A,
                                                const I shiftA,
                                                const I lda,
                                                const Istride strideA,
                                                const S abstol,
                                                S* residual,
                                                const I max_sweeps,
                                                I* n_sweeps,
                                                S* W,
                                                const Istride strideW,
                                                I* info,
                                                const I batch_count,
                                                T* Acpy,
                                                T* J,
                                                T* dwork,
                                                size_t size_dwork)
{
    ROCSOLVER_ENTER("rsyevj_heevj_template", "esort:", esort, "evect:", evect, "uplo:", uplo,
                    "n:", n, "shiftA:", shiftA, "lda:", lda, "abstol:", abstol,
                    "max_sweeps:", max_sweeps, "bc:", batch_count);

    // quick return
    bool const has_work = (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return rocblas_status_success;
    }

    auto const Atmp = Acpy;
    auto const shiftAtmp = 0 * shiftA;
    Istride const strideAtmp = n * n;
    auto const ldatmp = n;

    auto const Vtmp = J;
    auto const shiftVtmp = 0 * shiftA;
    Istride const strideVtmp = n * n;
    auto const ldvtmp = n;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threadsReset(BS1, 1, 1);

    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, residual, batch_count, 0);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, n_sweeps, batch_count, 0);
    ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, info, batch_count, 0);

    // scalar case
    if(n == 1)
    {
        ROCSOLVER_LAUNCH_KERNEL(scalar_case<T>, gridReset, threadsReset, 0, stream, evect, A,
                                strideA, W, strideW, batch_count);
    }

    // quick return
    if(n <= 1)
    {
        return (rocblas_status_success);
    }

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto is_even = [](auto n) { return ((n % 2) == 0); };
    auto is_odd = [](auto n) { return (!is_even(n)); };

    std::byte* pfree = (std::byte*)dwork;

    // absolute tolerance for evaluating when the algorithm has converged
    S const eps = get_epsilon<S>();
    S const atol = (abstol <= 0 ? eps : abstol);

    // local variables
    I const even_n = n + (n % 2);
    I const n_even = even_n;
    I const half_n = even_n / 2;

    bool const rsyevj_need_vector = true;
    I const nb = get_nb(n, rsyevj_need_vector);

    I const nblocks = ceil(n, nb);
    I const nblocks_even = nblocks + (nblocks % 2);

    bool const need_V = (evect != rocblas_evect_none);

    auto setup_schedule = [&](I const nplayers_arg, std::vector<I>& h_schedule, I* d_schedule) {
        // --------------------------------------------
        // generate schedule for even number of players
        // but skip over extra player
        // --------------------------------------------
        auto const nplayers = (nplayers_arg + (nplayers_arg % 2));

        {
            generateTournamentSequence(nplayers, h_schedule);

            // ------------------------------------
            // double check the schedule is correct
            // ------------------------------------
            assert(check_schedule<I>(nplayers, h_schedule));
        }

        {
            void* const dst = (void*)d_schedule;
            void* const src = (void*)&(h_schedule[0]);
            size_t const nbytes = sizeof(I) * nplayers * (nplayers - 1);

            HIP_CHECK(hipMemcpyAsync(dst, src, nbytes, hipMemcpyHostToDevice, stream));
        }
    };

    if(n <= RSYEVJ_BLOCKED_SWITCH(T, need_V))
    {
        // *** USE SINGLE SMALL-SIZE KERNEL ***
        // (TODO: RSYEVJ_BLOCKED_SWITCH may need re-tuning
        dim3 grid(1, 1, batch_count);

        I ddx = 32;
        I ddy = RSYEVJ_BDIM / ddx;

        auto const n_even = n + (n % 2);
        std::vector<I> h_schedule_small(n_even * (n_even - 1));

        std::byte* pfree = (std::byte*)dwork;
        I* const d_schedule_small = (I*)pfree;
        pfree += (n_even * (n_even - 1));

        setup_schedule(even_n, h_schedule_small, d_schedule_small);

        {
            size_t const lmemsize = 64 * 1024;
            ROCSOLVER_LAUNCH_KERNEL(rsyevj_small_kernel<T>, dim3(1, 1, batch_count),
                                    dim3(ddx, ddy, 1), lmemsize, stream, esort, evect, uplo, n, A,
                                    shiftA, lda, strideA, atol, eps, residual, max_sweeps, n_sweeps,
                                    W, strideW, info, Acpy, batch_count, d_schedule_small);
        }
    }
    else
    {
        // ------------------------
        // determine block size "nb" and
        // number of blocks "nblocks"
        // ------------------------
        bool const rsyevj_need_V = true;
        auto const nb = get_nb(n, rsyevj_need_V);
        auto const nblocks = ceil(n, nb);
        assert(is_even(nblocks));

        I const even_nb = nb + (nb % 2);
        I const even_nblocks = nblocks + (nblocks % 2);
        I const nblocks_half = even_nblocks / 2;

        I const nb_last = n - (nblocks - 1) * nb;
        assert(nb_last >= 1);

        std::vector<I> h_schedule_small((2 * nb) * ((2 * nb) - 1));
        std::vector<I> h_schedule_large(even_nblocks * (even_nblocks - 1));

        auto const num_rounds = (even_nblocks - 1);

        auto const shift_zero = 0 * shiftA;

        // --------------------------------------
        // preallocate storage for pointer arrays
        // --------------------------------------

        size_t total_bytes = 0;

        size_t const size_completed = sizeof(I) * (batch_count + 1);

        total_bytes += size_completed;

        size_t const size_Vj_bytes = sizeof(T) * (nb * 2) * (nb * 2) * (nblocks_half)*batch_count;
        size_t const size_Aj_bytes = sizeof(T) * (nb * 2) * (nb * 2) * (nblocks_half)*batch_count;

        total_bytes += size_Vj_bytes + size_Aj_bytes;

        size_t const size_Vj_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Aj_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;

        total_bytes += size_Vj_ptr_array + size_Aj_ptr_array;

        size_t const size_Vj_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Aj_last_ptr_array = sizeof(T*) * 1 * batch_count;

        total_bytes += size_Vj_last_ptr_array + size_Aj_last_ptr_array;

        size_t const size_A_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_row_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_A_col_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_A_ptr_array = sizeof(T*) * batch_count;

        total_bytes += size_A_row_ptr_array + size_A_col_ptr_array + size_A_row_last_ptr_array
            + size_A_col_last_ptr_array + size_A_ptr_array;

        size_t const size_Atmp_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Atmp_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Atmp_row_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Atmp_col_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Atmp_ptr_array = sizeof(T*) * batch_count;

        total_bytes += size_Atmp_row_ptr_array + size_Atmp_col_ptr_array
            + size_Atmp_row_last_ptr_array + size_Atmp_col_last_ptr_array + size_Atmp_ptr_array;

        size_t const size_Vtmp_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Vtmp_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Vtmp_row_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Vtmp_col_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Vtmp_ptr_array = sizeof(T) * batch_count;

        total_bytes += size_Vtmp_row_ptr_array + size_Vtmp_col_ptr_array
            + size_Vtmp_row_last_ptr_array + size_Vtmp_col_last_ptr_array + size_Vtmp_ptr_array;

        size_t const size_A_diag_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_last_diag_ptr_array = sizeof(T*) * 1 * batch_count;

        total_bytes += size_A_diag_ptr_array + size_A_last_diag_ptr_array;

        size_t const size_Gmat = sizeof(S) * (nblocks * nblocks) * batch_count;
        size_t const size_Gmat_ptr_array = sizeof(S*) * batch_count;
        size_t const size_Gmat_norm = sizeof(S) * batch_count;
        size_t const size_Amat_norm = sizeof(S) * batch_count;

        total_bytes += size_Gmat + size_Gmat_ptr_array;
        total_bytes += size_Gmat_norm + size_Amat_norm;

        size_t const size_schedule_small = sizeof(I) * (2 * nb) * ((2 * nb) - 1);
        size_t const size_schedule_large = sizeof(I) * nblocks_even * (nblocks_even - 1);

        std::byte* pfree = (std::byte*)dwork;

        I* const completed = (I*)pfree;
        pfree += size_completed;

        T* const Vj = (T*)pfree;
        pfree += size_Vj_bytes;
        T* const Aj = (T*)pfree;
        pfree += size_Aj_bytes;

        I const ldvj = (2 * nb);
        Istride const strideVj = (2 * nb) * (2 * nb);
        auto const shiftVj = shift_zero;

        I const ldaj = (2 * nb);
        Istride const strideAj = (2 * nb) * (2 * nb);
        auto const shiftAj = shift_zero;

        T** const Vj_ptr_array = (T**)pfree;
        pfree += size_Vj_ptr_array;
        T** const Aj_ptr_array = (T**)pfree;
        pfree += size_Aj_ptr_array;

        T** const Vj_last_ptr_array = (T**)pfree;
        pfree += size_Vj_last_ptr_array;
        T** const Aj_last_ptr_array = (T**)pfree;
        pfree += size_Aj_last_ptr_array;

        T** const A_row_ptr_array = (T**)pfree;
        pfree += size_A_row_ptr_array;
        T** const A_col_ptr_array = (T**)pfree;
        pfree += size_A_col_ptr_array;
        T** const A_row_last_ptr_array = (T**)pfree;
        pfree += size_A_row_last_ptr_array;
        T** const A_col_last_ptr_array = (T**)pfree;
        pfree += size_A_col_last_ptr_array;
        T** const A_ptr_array = (T**)pfree;
        pfree += size_A_ptr_array;

        T** const Atmp_row_ptr_array = (T**)pfree;
        pfree += size_Atmp_row_ptr_array;
        T** const Atmp_col_ptr_array = (T**)pfree;
        pfree += size_Atmp_col_ptr_array;
        T** const Atmp_row_last_ptr_array = (T**)pfree;
        pfree += size_Atmp_row_last_ptr_array;
        T** const Atmp_col_last_ptr_array = (T**)pfree;
        pfree += size_Atmp_col_last_ptr_array;
        T** const Atmp_ptr_array = (T**)pfree;
        pfree += size_Atmp_ptr_array;

        T** const Vtmp_row_ptr_array = (T**)pfree;
        pfree += size_Vtmp_row_ptr_array;
        T** const Vtmp_col_ptr_array = (T**)pfree;
        pfree += size_Vtmp_col_ptr_array;
        T** const Vtmp_row_last_ptr_array = (T**)pfree;
        pfree += size_Vtmp_row_last_ptr_array;
        T** const Vtmp_col_last_ptr_array = (T**)pfree;
        pfree += size_Vtmp_col_last_ptr_array;
        T** const Vtmp_ptr_array = (T**)pfree;
        pfree += size_Vtmp_ptr_array;

        T** const A_diag_ptr_array = (T**)pfree;
        pfree += size_A_diag_ptr_array;
        T** const A_last_diag_ptr_array = (T**)pfree;
        pfree += size_A_last_diag_ptr_array;

        S* const Gmat = (S*)pfree;
        pfree += size_Gmat;
        S** const Gmat_ptr_array = (S**)pfree;
        pfree += size_Gmat_ptr_array;
        S* const Gmat_norm = (S*)pfree;
        pfree += size_Gmat_norm;
        S* const Amat_norm = (S*)pfree;
        pfree += size_Amat_norm;

        I* const d_schedule_small = (I*)pfree;
        pfree += size_schedule_small;
        I* const d_schedule_large = (I*)pfree;
        pfree += size_schedule_large;

        char const c_uplo = (uplo == rocblas_fill_upper) ? 'U'
            : (uplo == rocblas_fill_lower)               ? 'L'
                                                         : 'A';
        // ----------------------------------
        // precompute norms of orginal matrix
        // ----------------------------------
        {
            // ---------------------------
            // make matrix to be symmetric
            // ---------------------------
            auto const max_blocks = 64 * 1000;
            auto const nx = 32;
            auto const ny = RSYEVJ_BDIM / nx;

            auto const nbx = ceil(n, nx);
            auto const nby = ceil(n, ny);
            auto const nbz = std::min(batch_count, max_blocks);

            ROCBLAS_LAUNCH_KERNEL((symmetrize_matrix_kernel<T, I, U, Istride>), dim3(nbx, nby, nbz),
                                  dim3(nx, ny, 1), 0, stream, c_uplo, n, A, shiftA, lda, strideA,
                                  batch_count);

            bool const need_diagonal = true;
            ROCBLAS_LAUNCH_KERNEL((cal_Gmat_kernel<T, I, S, Istride, U>), dim3(nbx, nby, nbz),
                                  dim3(nx, ny, 1), 0, stream,

                                  n, nb, A, shiftA, lda, strideA, Gmat, need_diagonal, batch_count);

            ROCBLAS_LAUNCH_KERNEL((sum_Gmat<S, I>), dim3(1, 1, nbz), dim3(nx, ny, 1), sizeof(S),
                                  stream, Gmat, Gmat_norm, batch_count);

            HIP_CHECK(hipMemcpy((void*)Amat_norm, (void*)Gmat_norm, sizeof(S) * batch_count, stream));
        }

        I n_completed = 0;
        I h_sweeps = 0;
        bool is_converged = false;

        for(; h_sweeps < max_sweeps; h_sweeps++)
        {
            {
                // compute norms of off diagonal blocks
                // setup completed[] array

                // -----------------------------------------------------
                // compute norms of blocks into array
                //
                // Gmat(0:(nblocks-1), 0:(nblocks-1), 0:(batch_count-1))
                // -----------------------------------------------------
                I const max_blocks = 64 * 1024;
                auto const nx = 32;
                auto const ny = RSYEVJ_BDIM / nx;

                auto const nbx = std::min(max_blocks, ceil(n, nx));
                auto const nby = std::min(max_blocks, ceil(n, ny));
                auto const nbz = std::min(max_blocks, batch_count);

                bool const need_diagonal = false;
                ROCBLAS_LAUNCH_KERNEL((cal_Gmat_kernel<T, I, S, Istride, U>), dim3(nbx, nby, nbz),
                                      dim3(nx, ny, 1), 0, stream,

                                      n, nb, A, shiftA, lda, strideA, Gmat, need_diagonal,
                                      batch_count);

                ROCBLAS_LAUNCH_KERNEL((sum_Gmat<S, I>), dim3(1, 1, nbz), dim3(nx, ny, 1), sizeof(S),
                                      stream, Gmat, Gmat_norm, batch_count);

                HIP_CHECK(hipMemsetAsync(&(completed[0]), 0, sizeof(I), stream));

                auto const nnx = 64;
                auto const nnb = ceil(batch_count, nnx);
                ROCBLAS_LAUNCH_KERNEL((set_completed<S, I, Istride>), dim3(nnb, 1, 1),
                                      dim3(nnx, 1, 1), 0, stream, n, nb, Amat_norm, abstol,
                                      Gmat_norm, completed, batch_count);
            }

            {
                // --------------------------------------
                // check convergence of all batch entries
                // --------------------------------------
                void* dst = (void*)n_completed;
                void* src = (void*)&(completed[0]);
                size_t const nbytes = sizeof(I);
                HIP_CHECK(hipMemcpyAsync(dst, src, nbytes, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));

                is_converged = (n_completed == batch_count);
                if(is_converged)
                {
                    break;
                };
            }

            auto const batch_count_remain = batch_count - n_completed;
            {
                // ------------------------------------------
                // build pointer arrays for data movement and
                // for rocblas batch GEMM operations
                // ------------------------------------------

                // ---------------------------------
                // reset value to be used as counter
                // ---------------------------------
                HIP_CHECK(hipMemsetAsync((void*)&(completed[0]), 0, sizeof(I), stream));

                auto const nx = 32;
                ROCSOLVER_LAUNCH_KERNEL(
                    (setup_ptr_arrays_kernel<T, I>), dim3(1, 1, batch_count), dim3(nx, 1, 1), 0,
                    stream,

                    n, nb, batch_count,

                    A, strideA, lda, shiftA, Atmp, strideAtmp, ldatmp, shiftAtmp, Vtmp, strideVtmp,
                    ldvtmp, shiftVtmp, completed,

                    Aj, Vj,

                    Vj_ptr_array, Aj_ptr_array, Vj_last_ptr_array, Aj_last_ptr_array,

                    A_row_ptr_array, A_col_ptr_array, A_row_last_ptr_array, A_col_last_ptr_array,
                    A_ptr_array,

                    Atmp_row_ptr_array, Atmp_col_ptr_array, Atmp_row_last_ptr_array,
                    Atmp_col_last_ptr_array, Atmp_ptr_array,

                    Vtmp_row_ptr_array, Vtmp_col_ptr_array, Vtmp_row_last_ptr_array,
                    Vtmp_col_last_ptr_array, Vtmp_ptr_array,

                    A_diag_ptr_array, A_last_diag_ptr_array, batch_count);
            }

            for(I iround = 0; iround < num_rounds; iround++)
            {
                // ------------------------
                // reorder and copy to Atmp, Vtmp
                // ------------------------
                I const* const col_map = d_schedule_large + iround * (even_nblocks);
                I const* const row_map = col_map;

                auto const max_blocks = 64 * 1000;
                auto const nx = 32;
                auto const ny = RSYEVJ_BDIM / nx;

                auto const nbx = std::min(max_blocks, ceil(n, nx));
                auto const nby = std::min(max_blocks, ceil(n, ny));
                auto const nbz = std::min(max_blocks, batch_count_remain);

                {
                    char const c_direction = 'F';

                    if(need_V)
                    {
                        // ------------------------------------
                        // matrix V need only column reordering
                        // ------------------------------------
                        I const* const null_row_map = nullptr;

                        ROCBLAS_LAUNCH_KERNEL((reorder_kernel<T, I, Istride>), dim3(nbx, nby, nbz),
                                              dim3(nx, ny, 1), 0, stream, c_direction, n, nb,
                                              null_row_map, col_map, Vtmp, shiftVtmp, ldvtmp,
                                              strideVtmp, Atmp, shiftAtmp, ldatmp, strideAtmp,
                                              batch_count_remain);

                        swap(Atmp, Vtmp);
                    }

                    ROCBLAS_LAUNCH_KERNEL((reorder_kernel<T, I, Istride>), dim3(nbx, nby, nbz),
                                          dim3(nx, ny, 1), 0, stream, c_direction, n, nb, row_map,
                                          col_map, A, shiftA, lda, strideA, Atmp, shiftAtmp, ldatmp,
                                          strideAtmp, batch_count_remain);
                }

                {// ------------------------------------------------------
                 // perform Jacobi iteration on independent sets of blocks
                 // ------------------------------------------------------

                 {// --------------------------
                  // copy diagonal blocks to Aj
                  // --------------------------

                  I const m1 = (2 * nb);
                I const n1 = (2 * nb);
                ROCBLAS_LAUNCH_KERNEL((lacpy_kernel<T, I, T**, T**, Istride>), dim3(nbx, nby, nbz),
                                      dim3(nx, ny, 1), 0, stream, c_uplo, m1, n1, A_diag_ptr_array,
                                      shiftA, lda, strideA, Aj_ptr_array, shiftAj, ldaj, strideAj,
                                      (nblocks_half - 1) * batch_count_remain);

                I const m2 = (nb + nb_last);
                I const n2 = (nb + nb_last);
                ROCBLAS_LAUNCH_KERNEL((lacpy_kernel<T, I, T**, T**, Istride>), dim3(nbx, nby, nbz),
                                      dim3(nx, ny, 1), 0, stream, c_uplo, m2, n2,
                                      A_last_diag_ptr_array, shiftA, lda, strideA, Aj_last_ptr_array,
                                      shiftAj, ldaj, strideAj, batch_count_remain);
            }

            {
                // -------------------------------------------------------
                // perform Jacobi iteration on small diagonal blocks in Aj
                // -------------------------------------------------------

                size_t const lmemsize = 64 * 1024;
                I const nn = (2 * nb);
                I const rsyevj_max_sweeps = 15;
                auto const rsyevj_atol = atol / nblocks;

                rocblas_evect const rsyevj_evect = rocblas_evect_original;

                // ----------------------------
                // set Vj to be diagonal matrix
                // ----------------------------

                {
                    char const c_uplo = 'A';
                    I const mm = (2 * nb);
                    I const nn = (2 * nb);

                    T alpha_offdiag = 0;
                    T beta_diag = 1;

                    ROCSOLVER_LAUNCH_KERNEL((laset_kernel<T, I, U, Istride>), dim3(nbx, nby, nbz),
                                            dim3(nx, ny, 1), 0, stream, c_uplo, mm, nn,
                                            alpha_offdiag, beta_diag, Aj, shiftAj, ldaj, strideAj,
                                            (nblocks_half)*batch_count_remain);
                }

                ROCSOLVER_LAUNCH_KERNEL(rsyevj_small_kernel<T>, dim3(1, 1, nbz), dim3(nx, ny, 1),
                                        lmemsize, stream, esort, rsyevj_evect, uplo, nn,
                                        Aj_ptr_array, shiftAj, ldaj, strideA, rsyevj_atol, eps,
                                        residual, rsyevj_max_sweeps, n_sweeps, W, strideW, info, Acpy,
                                        (nblocks_half - 1) * batch_count_remain, d_schedule_small);

                ROCSOLVER_LAUNCH_KERNEL(rsyevj_small_kernel<T>, dim3(1, 1, nbz), dim3(nx, ny, 1),
                                        lmemsize, stream, esort, rsyevj_evect, uplo, nn,
                                        Aj_last_ptr_array, shiftAj, ldaj, strideA, rsyevj_atol, eps,
                                        residual, rsyevj_max_sweeps, n_sweeps, W, strideW, info,
                                        Acpy, batch_count_remain, d_schedule_small);
            }
        }

        {
            // launch batch list to perform Vj' to update block rows
            // launch batch list to perform Vj' to update last block rows

            T alpha = 1;
            T beta = 0;
            auto m1 = 2 * nb;
            auto n1 = n;
            auto k1 = 2 * nb;
            ROCSOLVER_KERNEL_LAUNCH(rocblasCall_gemm(
                handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, m1, n1, k1,
                &alpha, Vj, shift_zero, ldvj, strideVj, Atmp_row_ptr_array, shift_zero, ldatmp,
                strideAtmp, &beta, A_row_ptr_array, shift_zero, lda, strideA,
                (nblocks_half - 1) * batch_count_remain));

            auto m2 = n;
            auto n2 = nb + nb_last;
            auto k2 = nb + nb_last;
            ROCBLAS_STATUS(rocblasCall_gemm(
                handle, rocblas_operation_conjugate_transpose, rocblas_operation_none, m2, n2, k2,
                &alpha, Vj, shiftVj, ldvj, strideVj, Atmp_row_ptr_array, shiftAtmp, ldatmp,
                strideAtmp, &beta, A_row_last_ptr_array, strideA, lda, strideA, batch_count_remain));
        }

        {
            // -------------------------------------------------------
            // launch batch list to perform Vj to update block columns
            // -------------------------------------------------------

            T alpha = 1;
            T beta = 0;
            I m1 = n;
            I n1 = 2 * nb;
            I k1 = 2 * nb;

            ROCBLAS_STATUS(rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none,
                                            m1, n1, k1, &alpha, Vj_ptr_array, strideVj, ldvj,
                                            strideVj, A_col_ptr_array, strideA, lda, strideA, &beta,
                                            Atmp_col_ptr_array, strideAtmp, ldatmp, strideAtmp,
                                            (nblocks_half - 1) * batch_count_remain));

            // -----------------------------------------------------------
            // launch batch list to perform Vj to update last block column
            // -----------------------------------------------------------

            I m2 = n;
            I n2 = nb_last;
            I k2 = nb_last;

            ROCBLAS_STATUS(rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none,
                                            m2, n2, k2, Vj_last_ptr_array, strideVtmp, ldvtmp,
                                            strideVtmp, A_col_last_ptr_array, strideA, lda, strideA,
                                            &beta, Atmp_col_last_ptr_array, strideAtmp, ldatmp,
                                            strideAtmp, batch_count_remain));

            {
                // -------------------
                // copy Atmp back to A
                // and undo reordering while copying
                // -----------------------------
                char const c_direction = 'B';

                ROCBLAS_LAUNCH_KERNEL((reorder_kernel<T, I, Istride>), dim3(nbx, nby, nbz),
                                      dim3(nx, ny, 1), 0, stream, c_direction, n, nb, row_map,
                                      col_map,

                                      Atmp_ptr_array, shiftAtmp, ldatmp, strideAtmp, A_ptr_array,
                                      shiftA, lda, strideA, batch_count_remain);
            }
        }

        if(need_V)
        {
            // launch batch list to perform Vj to update block columns for eigen vectors
            // launch batch list to perform Vj to update last block columns for eigen vectors

            T alpha = 1;
            T beta = 0;

            I m1 = n;
            I n1 = (2 * nb);
            I k1 = (2 * nb);
            ROCBLAS_STATUS(rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none,
                                            m1, n1, k1, &alpha, Vj_ptr_array, strideVj, ldvj,
                                            strideVj, Vtmp_col_ptr_array, strideVtmp, ldvtmp,
                                            strideVtmp, &beta, Atmp_col_ptr_array, strideAtmp,
                                            ldatmp, strideAtmp, (nblocks / 2 - 1) * batch_count));

            I m2 = n;
            I n2 = nb + nb_last;
            I k2 = nb + nb_last;
            ROCBLAS_STATUS(rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none,
                                            m2, n2, k2, &alpha, Vj_last_ptr_array, strideVj, ldvj,
                                            strideVj, Vtmp_col_last_ptr_array, strideVtmp, ldvtmp,
                                            strideVtmp, &beta, Atmp_col_last_ptr_array, strideAtmp,
                                            ldatmp, strideAtmp, batch_count));

            swap(Atmp, Vtmp);
        }

    } // end for iround

} // end for sweeps

{
    // ---------------------
    // copy out eigen values
    // ---------------------

    auto const max_blocks = 64 * 1000;
    auto const nx = 64;
    auto const nbx = std::min(max_blocks, ceil(n, nx));
    auto const nbz = std::min(max_blocks, batch_count);

    ROCBLAS_LAUNCH_KERNEL((copy_diagonal_kernel<T, I, U, Istride>), dim3(1, 1, nbz), dim3(nx, 1, 1),
                          0, stream, n, A, shiftA, lda, strideA, W, strideW, batch_count);
}

{
    // -----------------------------------------------
    // over-write original matrix A with eigen vectors
    // -----------------------------------------------
    if(need_V)
    {
        lacpy(handle, n, n, Vtmp, shiftVtmp, ldvtmp, strideVtmp, A, shiftA, lda, strideA,
              batch_count);
    }
}

} // end large block
}

#if(0)

bool ev = (evect != rocblas_evect_none);
I h_sweeps = 0;
I h_completed = 0;

// set completed = 0
ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threadsReset, 0, stream, completed, batch_count + 1, 0);

// copy A to Acpy, set A to identity (if applicable), compute initial residual, and
// initialize top/bottom pairs (if applicable)
ROCSOLVER_LAUNCH_KERNEL(syevj_init<T>,
                        grid,
                        threads,
                        lmemsizeInit,
                        stream,
                        evect,
                        uplo,
                        half_blocks,
                        n,
                        A,
                        shiftA,
                        lda,
                        strideA,
                        atol,
                        residual,
                        Acpy,
                        norms,
                        top,
                        bottom,
                        completed);

while(h_sweeps < max_sweeps)
{
    // if all instances in the batch have finished, exit the loop
    HIP_CHECK(hipMemcpyAsync(&h_completed, completed, sizeof(I), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    if(h_completed == batch_count)
        break;

    // decompose diagonal blocks
    ROCSOLVER_LAUNCH_KERNEL(syevj_diag_kernel<T>, gridDK, threadsDK, lmemsizeDK, stream, n, Acpy, 0,
                            n, n * n, eps, J, completed);

    // apply rotations calculated by diag_kernel
    ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<false, T, S>), gridDR, threadsDR, lmemsizeDR, stream,
                            true, n, Acpy, 0, n, n * n, J, completed);
    ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<true, T, S>), gridDR, threadsDR, lmemsizeDR, stream,
                            true, n, Acpy, 0, n, n * n, J, completed);

    // update eigenvectors
    if(ev)
        ROCSOLVER_LAUNCH_KERNEL((syevj_diag_rotate<false, T, S>), gridDR, threadsDR, lmemsizeDR,
                                stream, false, n, A, shiftA, lda, strideA, J, completed);

    if(half_blocks == 1)
    {
        // decompose off-diagonal block
        ROCSOLVER_LAUNCH_KERNEL((syevj_offd_kernel<T, S>), gridOK, threadsOK, lmemsizeOK, stream,
                                blocks, n, Acpy, 0, n, n * n, eps, (ev ? J : nullptr), top, bottom,
                                completed);

        // update eigenvectors
        if(ev)
            ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR, threadsOR, 0, stream,
                                    false, blocks, n, A, shiftA, lda, strideA, J, top, bottom,
                                    completed);
    }
    else
    {
        for(I b = 0; b < even_blocks - 1; b++)
        {
            // decompose off-diagonal blocks, indexed by top/bottom pairs
            ROCSOLVER_LAUNCH_KERNEL((syevj_offd_kernel<T, S>), gridOK, threadsOK, lmemsizeOK, stream,
                                    blocks, n, Acpy, 0, n, n * n, eps, J, top, bottom, completed);

            // apply rotations calculated by offd_kernel
            ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR, threadsOR, 0, stream,
                                    true, blocks, n, Acpy, 0, n, n * n, J, top, bottom, completed);
            ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<true, T, S>), gridOR, threadsOR, 0, stream,
                                    true, blocks, n, Acpy, 0, n, n * n, J, top, bottom, completed);

            // update eigenvectors
            if(ev)
                ROCSOLVER_LAUNCH_KERNEL((syevj_offd_rotate<false, T, S>), gridOR, threadsOR, 0,
                                        stream, false, blocks, n, A, shiftA, lda, strideA, J, top,
                                        bottom, completed);

            // cycle top/bottom pairs
            ROCSOLVER_LAUNCH_KERNEL(syevj_cycle_pairs<T>, gridPairs, threads, lmemsizePairs, stream,
                                    half_blocks, top, bottom);
        }
    }

    // compute new residual
    h_sweeps++;
    ROCSOLVER_LAUNCH_KERNEL(syevj_calc_norm<T>, grid, threads, lmemsizeInit, stream, n, h_sweeps,
                            residual, Acpy, norms, completed);
}

// set outputs and sort eigenvalues & vectors
ROCSOLVER_LAUNCH_KERNEL(syevj_finalize<T>,
                        grid,
                        threads,
                        0,
                        stream,
                        esort,
                        evect,
                        n,
                        A,
                        shiftA,
                        lda,
                        strideA,
                        residual,
                        max_sweeps,
                        n_sweeps,
                        W,
                        strideW,
                        info,
                        Acpy,
                        completed);
}

return rocblas_status_success;
}
#endif

ROCSOLVER_END_NAMESPACE
