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
static constexpr int idebug = 1;
#define TRACE(ival)                                       \
    {                                                     \
        auto istat = hipDeviceSynchronize();              \
        if(idebug >= ival)                                \
        {                                                 \
            printf("trace: %s:%d\n", __FILE__, __LINE__); \
            fflush(stdout);                               \
        }                                                 \
    }

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

// ----------------------------------------------
// CPU code to adjust the schedule due to
// permutation in each round so that the independent
// pairs are contiguous (0,1), (2,3), ...
// ----------------------------------------------
template <typename I>
static void adjust_schedule(I const nplayers, std::vector<I>& schedule)
{
    I const num_rounds = (nplayers - 1);
    assert(schedule.size() >= (num_rounds * nplayers));

    for(I iround = 0; iround < num_rounds; iround++)
    {
        std::vector<I> new2old(nplayers);
        std::vector<I> old2new(nplayers);

        // ---------------------------
        // form new2old(:) permutation
        // ---------------------------
        I* const cp = &(schedule[iround * nplayers]);
        for(I i = 0; i < nplayers; i++)
        {
            new2old[i] = cp[i];
        }

        // ---------------------------
        // form new2old(:) permutation
        // ---------------------------
        for(I i = 0; i < nplayers; i++)
        {
            old2new[new2old[i]] = i;
        }

        // -------------------------
        // update remaining schedule
        // -------------------------
        for(I jround = iround + 1; jround < num_rounds; jround++)
        {
            I* const rp = &(schedule[jround * nplayers]);
            for(auto i = 0; i < nplayers; i++)
            {
                auto const iold = rp[i];
                auto const inew = old2new[iold];
                rp[i] = inew;
            }
        }
    }
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

static constexpr auto NX_THREADS = 32;

// Max number of threads per thread-block used in rsyevj_small kernel
static constexpr auto RSYEVJ_BDIM = 1024;

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
            bool const is_same = (aij == std::conj(aji));
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

    auto A = [=](auto i, auto j) -> T& { return (A_[idx2D(i, j, lda)]); };

    for(auto j = j_start; j < n; j += j_inc)
    {
        for(auto i = i_start; i < n; i += i_inc)
        {
            bool const is_strictly_lower = (i > j);
            bool const is_strictly_upper = (i < j);
            bool const is_diag = (i == j);

            bool const do_assignment
                = (use_upper && is_strictly_upper) || (use_lower && is_strictly_lower);
            if(do_assignment)
            {
                A(j, i) = conj(A(i, j));
            }

            if(is_diag)
            {
                auto const aii = A(i, i);
                A(i, i) = (aii + conj(aii)) / 2;
            }
        }
    }

    __syncthreads();

#ifdef NDEBUG
#else
    {
        // --------------------------------
        // double check matrix is symmetric
        //
        // note this operation may involve multiple thread blocks
        // --------------------------------
        bool is_symmetric = true;

        T aij_err = 0;
        T aji_err = 0;

        I ierr = 0;
        I jerr = 0;

        for(auto j = j_start; (j < n) && is_symmetric; j += j_inc)
        {
            for(auto i = i_start; (i < n) && is_symmetric; i += i_inc)
            {
                bool const is_strictly_lower = (i > j);
                bool const is_strictly_upper = (i < j);
                bool const is_diag = (i == j);
                bool const do_check = (use_upper && is_strictly_upper)
                    || (use_lower && is_strictly_lower) || is_diag;
                if(do_check)
                {
                    auto const aij = A(i, j);
                    auto const aji = A(j, i);
                    bool const is_same = (aji == conj(aij));
                    if(!is_same)
                    {
                        aij_err = aij;
                        aij_err = aji;
                        ierr = i;
                        jerr = j;
                        is_symmetric = false;
                        break;
                    }
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
                                                Istride const shiftA,
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
// if (uplo == 'F') set only the off-diagonal part
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
                                    Istride const shiftA,
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

        laset_body(c_uplo, m, n, alpha_offdiag, beta_diag, A_p, lda, i_start, i_inc, j_start, j_inc);
    }
}

/************** Kernels and device functions for large size*******************/
/*****************************************************************************/

/**   sort eigen values in array W(:)
 *
 *  launch as dim3(1,1,nbz), dim3(nx,1,1)
 */
template <typename S, typename I, typename Istride>
__global__ static void sort_kernel(I const n,
                                   S* const W_,
                                   Istride const strideW,
                                   I* const map_,
                                   Istride const stridemap,
                                   I const batch_count)
{
    bool const has_map = (map_ != nullptr);

    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        S* W = W_ + bid * strideW;
        I* map = (has_map) ? map_ + stridemap : nullptr;

        shell_sort(n, W, map);
    }
}

/** perform gather operations

    B(i,j) = A( row_map[i], col_map[j])

    launch as dim3(nbx,nby,nbz), dim3(nx,ny,1)
*/
template <typename T, typename I, typename Istride, typename UA, typename UB>
__global__ static void gather2D_kernel(I const m,
                                       I const n,
                                       I const* const row_map,
                                       I const* const col_map,
                                       UA A,
                                       Istride const shiftA,
                                       I const lda,
                                       Istride strideA,
                                       UB B,
                                       Istride const shiftB,
                                       I const ldb,
                                       Istride strideB,
                                       I const batch_count)
{
    bool const has_row_map = (row_map != nullptr);
    bool const has_col_map = (col_map != nullptr);

    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipBlockDim_z * hipGridDim_z;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;

    I const j_start = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;
    I const j_inc = hipBlockDim_y * hipGridDim_y;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const A_p = load_ptr_batch(A, bid, shiftA, strideA);
        T* const B_p = load_ptr_batch(B, bid, shiftB, strideB);

        auto const Ap = [=](auto i, auto j) -> const T& { return (A_p[idx2D(i, j, lda)]); };
        auto const Bp = [=](auto i, auto j) -> T& { return (B_p[idx2D(i, j, ldb)]); };

        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = i_start; i < m; i += i_inc)
            {
                auto const ia = has_row_map ? row_map[i] : i;
                auto const ja = has_col_map ? col_map[j] : j;

                Bp(i, j) = Ap(ia, ja);
            }
        }
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
                                      Istride const shiftA,
                                      I const ldA,
                                      Istride strideA,
                                      UC CC,
                                      Istride const shiftC,
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
    {
        bool const is_root = (i_start == 0) && (j_start == 0);
        if(has_row_map)
        {
            bool const is_row_ok = (row_map[(nblocks - 1)] == (nblocks - 1));
            if((!is_row_ok) && (is_root))
            {
                for(auto ib = 0; ib < nblocks; ib++)
                {
                    printf("row_map[%d] = %d\n", (int)ib, (int)row_map[ib]);
                }
            }
            assert(is_row_ok);
        }

        if(has_col_map)
        {
            bool const is_col_ok = (col_map[(nblocks - 1)] == (nblocks - 1));
            if((!is_col_ok) && (is_root))
            {
                for(auto ib = 0; ib < nblocks; ib++)
                {
                    printf("col_map[%d] = %d\n", (int)ib, (int)col_map[ib]);
                }
            }
            assert(col_map[(nblocks - 1)] == (nblocks - 1));
        }
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
template <typename T, typename I, typename AA, typename Istride, typename S>
__global__ static void copy_diagonal_kernel(I const n,
                                            AA A,
                                            Istride const shiftA,
                                            I const lda,
                                            Istride const strideA,

                                            S* const W,
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
        auto const W_p = W + bid * strideW;

        for(auto i = i_start; i < n; i += i_inc)
        {
            T const aii = A_p[idx2D(i, i, lda)];
            W_p[i] = std::real(aii);
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
                                    Istride const shiftA,
                                    I const lda,
                                    Istride strideA,
                                    CC C,
                                    Istride const shiftC,
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

        assert(Ap != nullptr);
        assert(Cp != nullptr);

        lacpy_body(uplo, m, n, Ap, lda, Cp, ldc, i_start, i_inc, j_start, j_inc);
    }
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
        printf("a = %le; b = %le; c = %le;\n", (double)a, (double)b, (double)c);
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
    if(idebug >= 2)
    {
        // --------------------
        // double check results
        // --------------------
        // [cs1  conj(sn1) ]  [ a        b]  [ cs1   -conj(sn1) ] = [ rt1    0  ]
        // [-sn1  cs1      ]  [ conj(b)  c]  [ sn1    cs1       ]   [ 0      rt2]

        // -------------------------------------------------
        // [cs1  conj(sn1) ]  [ a        b]  -> [a11   a12]
        // [-sn1  cs1      ]  [ conj(b)  c]     [a21   a22]
        // -------------------------------------------------
        auto const a11 = cs1 * a + conj(sn1) * conj(b);
        auto const a12 = cs1 * b + conj(sn1) * c;
        auto const a21 = (-sn1) * a + cs1 * conj(b);
        auto const a22 = (-sn1) * b + cs1 * c;

        // -----------------------------------------------
        // [a11 a12]  [ cs1   -conj(sn1) ] = [ rt1    0  ]
        // [a21 a22]  [ sn1    cs1       ]   [ 0      rt2]
        // -----------------------------------------------

        auto const anorm = std::sqrt(std::norm(a) + std::norm(b) * 2 + std::norm(c));
        auto const anorm_eig = std::sqrt(std::norm(rt1) + std::norm(rt2));

        auto const e11 = a11 * cs1 + a12 * sn1 - rt1;
        auto const e12 = a11 * (-conj(sn1)) + a12 * cs1;
        auto const e21 = a21 * cs1 + a22 * sn1;
        auto const e22 = a21 * (-conj(sn1)) + a22 * cs1 - rt2;

        auto const enorm
            = std::sqrt(std::norm(e11) + std::norm(e12) + std::norm(e21) + std::norm(e22));

        auto const tol = 1e-6;
        auto isok = (enorm <= tol * anorm);

        if(!isok)
        {
            printf("zlaev2: enorm=%le; anorm=%le; anorm_eig=%le;\n", (double)enorm, (double)anorm,
                   (double)anorm_eig);
            printf("a = %le+%le * i ; b = %le+ %le  * i; c = %le + %le *  i;\n", std::real(a),
                   std::imag(a), std::real(b), std::imag(b), std::real(c), std::imag(c));
            printf("abs(e11)=%le; abs(e12)=%le; abs(e21)=%le; abs(e22)=%le;\n", std::abs(e11),
                   std::abs(e12), std::abs(e21), std::abs(e22));
            printf("rt1 = %le; rt2 = %le;\n", (double)rt1, (double)rt2);
        }
        assert(isok);
    }
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

        if(dsum != zero)
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

#ifdef NDEBUG
#else
template <typename T, typename I>
static void check_ptr_array(std::string msg, I const n, T* d_ptr_array[])
{
    std::vector<T*> h_ptr_array(n);

    auto const istat
        = hipMemcpy(&(h_ptr_array[0]), d_ptr_array, sizeof(T*) * n, hipMemcpyDeviceToHost);
    assert(istat == HIP_SUCCESS);

    I nerr = 0;
    for(auto i = 0; i < n; i++)
    {
        if(h_ptr_array[i] == nullptr)
        {
            nerr++;
        }
    }
    if(nerr != 0)
    {
        printf("check_ptr_array:%s\n", msg.c_str());
        fflush(stdout);
    }
}
#endif

/** kernel to setup pointer arrays in preparation
 * for calls to batched GEMM and for copying data
 *
 * launch as dim3(1,1,batch_count), dim3(nx,1,1)
**/
template <typename T, typename I, typename Istride, typename AA, typename BB, typename CC>
__global__ static void setup_ptr_arrays_kernel(

    I const n,
    I const nb,

    AA A,
    Istride const shiftA,
    I const lda,
    Istride const strideA,

    BB Atmp,
    Istride const shiftAtmp,
    I const ldatmp,
    Istride const strideAtmp,

    CC Vtmp,
    Istride const shiftVtmp,
    I const ldvtmp,
    Istride const strideVtmp,

    T* const Aj,
    Istride const strideAj,
    T* const Vj,
    Istride const strideVj,
    T* const Aj_last,
    T* const Vj_last,

    I* const completed,

    T* Vj_ptr_array[],
    T* Aj_ptr_array[],
    T* Vj_last_ptr_array[],
    T* Aj_last_ptr_array[],

    T* A_row_ptr_array[],
    T* A_col_ptr_array[],
    T* A_last_row_ptr_array[],
    T* A_last_col_ptr_array[],
    T* A_ptr_array[],

    T* Atmp_row_ptr_array[],
    T* Atmp_col_ptr_array[],
    T* Atmp_last_row_ptr_array[],
    T* Atmp_last_col_ptr_array[],
    T* Atmp_ptr_array[],

    T* Vtmp_row_ptr_array[],
    T* Vtmp_col_ptr_array[],
    T* Vtmp_last_row_ptr_array[],
    T* Vtmp_last_col_ptr_array[],
    T* Vtmp_ptr_array[],

    T* A_diag_ptr_array[],
    T* A_last_diag_ptr_array[],

    I const batch_count)
{
    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_start = hipThreadIdx_x;
    I const i_inc = hipBlockDim_x;

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

        // auto const ibatch = atomicAdd(&(completed[0]), 1);
        atomicAdd(&(completed[0]), 1);
        auto const ibatch = bid;

        A_ptr_array[ibatch] = A_p;
        Atmp_ptr_array[ibatch] = Atmp_p;
        Vtmp_ptr_array[ibatch] = Vtmp_p;

        auto const ilast_row = (nblocks_half - 1) * (2 * nb);
        auto const jlast_col = (nblocks_half - 1) * (2 * nb);

        A_last_row_ptr_array[ibatch] = A_p + shiftA + idx2D(ilast_row, 0, lda);
        A_last_col_ptr_array[ibatch] = A_p + shiftA + idx2D(0, jlast_col, lda);

        A_last_diag_ptr_array[ibatch] = A_p + shiftA + idx2D(ilast_row, jlast_col, lda);

        Atmp_last_row_ptr_array[ibatch] = Atmp_p + shiftAtmp + idx2D(ilast_row, 0, ldatmp);
        Atmp_last_col_ptr_array[ibatch] = Atmp_p + shiftAtmp + idx2D(0, jlast_col, ldatmp);

        Vtmp_last_row_ptr_array[ibatch] = Vtmp_p + shiftVtmp + idx2D(ilast_row, 0, ldvtmp);
        Vtmp_last_col_ptr_array[ibatch] = Vtmp_p + shiftVtmp + idx2D(0, jlast_col, ldvtmp);

        {
            for(auto i = i_start; i < (nblocks_half - 1); i += i_inc)
            {
                auto const ip = i + ibatch * (nblocks_half - 1);
                Vj_ptr_array[ip] = Vj + bid * strideVj + i * (2 * nb) * (2 * nb);

                assert(Vj_ptr_array[ip] != nullptr);
            }

            Vj_last_ptr_array[ibatch] = Vj_last + ibatch * (2 * nb) * (2 * nb);

            assert(Vj_last_ptr_array[ibatch] != nullptr);
        }

        {
            for(auto i = i_start; i < (nblocks_half - 1); i += i_inc)
            {
                auto const ip = i + ibatch * (nblocks_half - 1);
                Aj_ptr_array[ip] = Aj + ibatch * strideAj + i * (2 * nb) * (2 * nb);

                assert(Aj_ptr_array[ip] != nullptr);
            }

            Aj_last_ptr_array[ibatch] = Aj_last + ibatch * ((2 * nb) * (2 * nb));

            assert(Aj_last_ptr_array[ibatch] != nullptr);
        }

        for(auto i = i_start; i < (nblocks_half - 1); i += i_inc)
        {
            auto const ip = i + ibatch * (nblocks_half - 1);

            {
                I const irow = i * (2 * nb);
                I const jcol = 0;
                A_row_ptr_array[ip] = A_p + shiftA + idx2D(irow, jcol, lda);
                Atmp_row_ptr_array[ip] = Atmp_p + shiftAtmp + idx2D(irow, jcol, ldatmp);
                Vtmp_row_ptr_array[ip] = Vtmp_p + shiftVtmp + idx2D(irow, jcol, ldvtmp);

                assert(A_row_ptr_array[ip] != nullptr);
                assert(Atmp_row_ptr_array[ip] != nullptr);
                assert(Vtmp_row_ptr_array[ip] != nullptr);
            }

            {
                auto const irow = 0;
                auto const jcol = i * (2 * nb);

                A_col_ptr_array[ip] = A_p + shiftA + idx2D(irow, jcol, lda);
                Atmp_col_ptr_array[ip] = Atmp_p + shiftAtmp + idx2D(irow, jcol, ldatmp);
                Vtmp_col_ptr_array[ip] = Vtmp_p + shiftVtmp + idx2D(irow, jcol, ldvtmp);

                assert(A_col_ptr_array[ip] != nullptr);
                assert(Atmp_col_ptr_array[ip] != nullptr);
                assert(Vtmp_col_ptr_array[ip] != nullptr);

                {
                    I const irow = (2 * nb) * i;
                    I const jcol = irow;
                    A_diag_ptr_array[ip] = A_p + shiftA + idx2D(irow, jcol, lda);

                    assert(A_diag_ptr_array[ip] != nullptr);
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
        return (schedule_[i1 + itable * 2 + iround * n_even]);
    };

    auto V = [=](auto i, auto j) -> T& { return (V_[i + j * ldv]); };
    auto A = [=](auto i, auto j) -> T& { return (A_[i + j * lda]); };

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

        laset_body<T, I>(c_uplo, mm, nn, alpha_offdiag, beta_diag, V_, ld1, i_start, i_inc, j_start,
                         j_inc);

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
    if(idebug >= 1)
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
        {
            S* const Swork = (S*)dwork;

            auto const mm = n;
            auto const nn = n;
            cal_norm_body(mm, nn, A_, lda, Swork, i_start, i_inc, j_start, j_inc, need_diagonal);
            norm_offdiag = Swork[0];
        }

        has_converged = (norm_offdiag <= abstol * norm_A);
        __syncthreads();

        if(has_converged)
        {
            break;
        };

        for(auto iround = 0; iround < num_rounds; iround++)
        {
            {
                // --------------------
                // precompute rotations
                // --------------------
                auto const ij_start = i_start + j_start * i_inc;
                auto const ij_inc = i_inc * j_inc;

                for(auto ij = ij_start; ij < ntables; ij += ij_inc)
                {
                    // --------------------------
                    // get independent pair (p,q)
                    // --------------------------

                    auto const p = schedule(0, ij, iround);
                    auto const q = schedule(1, ij, iround);

                    {
                        // -------------------------------------
                        // if n is an odd number, then just skip
                        // operation for invalid values
                        // -------------------------------------
                        bool const is_valid_pq = (0 <= p) && (p < n) && (0 <= q) && (q < n);

                        if(!is_valid_pq)
                        {
                            continue;
                        }
                    }

                    // ----------------------------------------------------------------------
                    // [ cs1  conj(sn1) ][ App        Apq ] [cs1   -conj(sn1)] = [rt1   0   ]
                    // [-sn1  cs1       ][ conj(Apq)  Aqq ] [sn1    cs1      ]   [0     rt2 ]
                    // ----------------------------------------------------------------------
                    {
                        T sn1 = 0;
                        S cs1 = 0;
                        S rt1 = 0;
                        S rt2 = 0;
                        auto const App = A(p, p);
                        auto const Apq = A(p, q);
                        auto const Aqq = A(q, q);
                        laev2<T, S>(App, Apq, Aqq, rt1, rt2, cs1, sn1);

                        cosine[ij] = cs1;
                        sine[ij] = sn1;
                    }
                } // end for ij

                __syncthreads();
            }

            // ---------------------------
            // apply row update operations
            // ---------------------------

            for(auto j = j_start; j < ntables; j += j_inc)
            {
                //  ----------------------------------
                //  We have
                //
                //  J' * [App  Apq] * J = [rt1   0  ]
                //       [Apq' Aqq]       [0     rt2]
                //
                //
                //  J = [cs1   -conj(sn1)]
                //      [sn1    cs1      ]
                //
                //  J' = [cs1    conj(sn1) ]
                //       [-sn1   cs1       ]
                //  ----------------------------------
                auto const p = schedule(0, j, iround);
                auto const q = schedule(1, j, iround);

                // -------------------------------------
                // if n is an odd number, then just skip
                // operation for invalid values
                // -------------------------------------
                bool const is_valid_pq = (0 <= p) && (p < n) && (0 <= q) && (q < n);

                if(!is_valid_pq)
                {
                    continue;
                }

                auto const cs1 = cosine[j];
                auto const sn1 = sine[j];

                // ------------------------
                // J' is conj(transpose(J))
                // ------------------------
                auto const Jt11 = cs1;
                auto const Jt12 = conj(sn1);
                auto const Jt21 = -sn1;
                auto const Jt22 = cs1;

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

                {
                    // -------------------------------------
                    // if n is an odd number, then just skip
                    // operation for invalid values
                    // -------------------------------------
                    bool const is_valid_pq = (0 <= p) && (p < n) && (0 <= q) && (q < n);

                    if(!is_valid_pq)
                    {
                        continue;
                    }
                }

                // ----------------------------------------------------------------------
                // [ cs1  conj(sn1) ][ App        Apq ] [cs1   -conj(sn1)] = [rt1   0   ]
                // [-sn1  cs1       ][ conj(Apq)  Aqq ] [sn1    cs1      ]   [0     rt2 ]
                // ----------------------------------------------------------------------

                auto const cs1 = cosine[j];
                auto const sn1 = sine[j];

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
                if(idebug >= 2)
                {
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
                }

#endif

                // -------------------------------------
                // explicitly set A(p,q), A(q,p) be zero
                // otherwise, abs(A(p,q)) might still be around
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

#ifdef NDEBUG
#else
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
#endif

    *info = (has_converged) ? 0 : 1;
    *n_sweeps = (has_converged) ? isweep : max_sweeps;
    *residual = norm_offdiag;

#ifdef NDEBUG
#else
    // debug
    if(idebug >= 1)
    {
        if((i_start == 0) && (j_start == 0))
        {
            printf("isweep=%d, abstol=%le,norm_offdiag = %le, norm_A = %le\n", (int)isweep,
                   (double)abstol, (double)norm_offdiag, (double)norm_A);
        }
    }
#endif

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

// -----------------------------------------
// compute the Frobenius norms of the blocks into
// Gmat(0:(nblocks-1),0:(nblocks-1),batch_count)
//
// launch as dim3(nbx,nby,nbz), dim3(nx,ny,1)
// -----------------------------------------
template <typename T, typename I, typename S, typename Istride, typename AA>
__global__ static void cal_Gmat_kernel(I const n,
                                       I const nb,
                                       AA A,
                                       Istride const shiftA,
                                       I const lda,
                                       Istride const strideA,

                                       S* const Gmat_,
                                       bool const include_diagonal_values,

                                       I const* const completed,
                                       I const batch_count)
{
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto const nblocks = ceil(n, nb);

    // ------------------------------------------------------
    // nb_last is the size of the last partially filled block
    // ------------------------------------------------------
    auto const nb_last = n - (nblocks - 1) * nb;
    assert((1 <= nb_last) && (nb_last <= nb));

    auto const idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto Gmat = [=](auto ib, auto jb, auto bid) -> S& {
        return (Gmat_[ib + jb * nblocks + bid * (nblocks * nblocks)]);
    };

    // -------------------------------------------
    // bsize(iblock)  computes the size of (iblock)
    // -------------------------------------------
    auto bsize = [=](auto iblock) { return ((iblock == (nblocks - 1)) ? nb_last : nb); };

    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const ib_start = hipBlockIdx_x;
    I const jb_start = hipBlockIdx_y;

    I const ib_inc = hipGridDim_x;
    I const jb_inc = hipGridDim_y;

    I const i_start = hipThreadIdx_x;
    I const j_start = hipThreadIdx_y;

    I const i_inc = hipBlockDim_x;
    I const j_inc = hipBlockDim_y;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        // ----------------------------
        // note use index value bid + 1
        // ----------------------------
        bool const is_completed = completed[bid + 1];
        if(is_completed)
        {
            continue;
        }

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

                // -----------------------------------------------
                // compute norm of (iblock,jblock) submatrix block
                // -----------------------------------------------
                cal_norm_body(ni, nj, &(Ap(ii, jj)), lda, &(Gmat(ib, jb, bid)), i_start, i_inc,
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
__global__ static void sum_Gmat(I const n,
                                I const nb,
                                S* const Gmat_,
                                S* const Gnorm_,
                                I const* const completed,
                                I const batch_count)
{
    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_start = hipThreadIdx_x;
    I const i_inc = hipBlockDim_x;

    I const j_start = hipThreadIdx_y;
    I const j_inc = hipBlockDim_y;

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };
    I const nblocks = ceil(n, nb);

    auto Gmat = [=](auto i, auto j, auto bid) -> const S& {
        return (Gmat_[i + j * nblocks + bid * (nblocks * nblocks)]);
    };

    auto Gnorm = [=](auto bid) -> S& { return (Gnorm_[bid]); };

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        // -----------------------------
        // note use index value (bid + 1)
        // -----------------------------
        bool const is_completed = completed[bid + 1];
        if(is_completed)
        {
            continue;
        };

        auto const mm = nblocks;
        auto const nn = nblocks;
        auto const ld1 = nblocks;

        bool const include_diagonal = true;
        cal_norm_body(mm, nn, &(Gmat(0, 0, bid)), ld1, &(Gnorm(bid)), i_start, i_inc, j_start,
                      j_inc, include_diagonal);
    }
}

// ------------------------------------------------------
// assume launch as   dim3(nbx,1,1), dim3(nx,1,1)
// ------------------------------------------------------
template <typename S, typename I, typename Istride>
__global__ static void set_completed_kernel(I const n,
                                            I const nb,
                                            S* const Anorm,
                                            S const abstol,

                                            I const h_sweeps,
                                            I n_sweeps[],
                                            S residual[],
                                            I info[],
                                            I completed[],
                                            I const batch_count)
{
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto const nblocks = ceil(n, nb);

    auto const bid_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    auto const bid_inc = hipBlockDim_x * hipGridDim_x;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        bool const is_already_completed = completed[bid + 1];
        if(is_already_completed)
        {
            continue;
        };

        S const anorm = Anorm[bid];
        S const gnorm = residual[bid];
        bool const is_completed = (gnorm <= abstol * anorm);

        // -----------------
        // note use "bid + 1"
        // -----------------
        completed[bid + 1] = is_completed;

        info[bid] = (is_completed) ? 0 : 1;

        if(is_completed)
        {
            n_sweeps[bid] = h_sweeps;

            atomicAdd(&(completed[0]), 1);
        };
    }
}

#if(0)
template <typename S, typename I>
__global__ static void set_info_kernel(I const max_sweeps, I n_sweeps[], I info[], I const batch_count)
{
    auto const bid_start = hipThreadIdx_z + hipBlockIdx_z * hipBlockIdx_z;
    auto const bid_inc = hipBlockDim_z * hipGridDim_z;

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        bool const is_converged = (n_sweeps[bid] < max_sweeps);

        info[bid] = (is_converged) ? 0 : 1;
    }
}
#endif
template <typename I>
static rocblas_status setup_schedule(I const nplayers_arg,
                                     std::vector<I>& h_schedule,
                                     I* d_schedule,
                                     hipStream_t stream,
                                     bool const use_adjust_schedule = false)
{
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

        if(use_adjust_schedule)
        {
            adjust_schedule(nplayers, h_schedule);
        }
    }

    {
        void* const dst = (void*)d_schedule;
        void* const src = (void*)&(h_schedule[0]);
        size_t const nbytes = sizeof(I) * nplayers * (nplayers - 1);

        HIP_CHECK(hipMemcpyAsync(dst, src, nbytes, hipMemcpyHostToDevice, stream));
    }

    return (rocblas_status_success);
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

template <typename T, typename I, typename S, typename U, typename Istride>
ROCSOLVER_KERNEL void __launch_bounds__(RSYEVJ_BDIM)
    rsyevj_small_kernel(const rocblas_esort esort,
                        const rocblas_evect evect,
                        const rocblas_fill uplo,
                        const I n,
                        U AA,
                        const I shiftA,
                        const I lda,
                        const Istride strideA,
                        const S abstol,
                        const S eps,
                        S* residualA,
                        const I max_sweeps,
                        I* n_sweepsA,
                        S* WW,
                        const Istride strideW,
                        I* infoA,
                        T* AcpyA,
                        const Istride strideAcpyA,

                        I batch_count,
                        I* schedule_,
                        bool const do_overwrite_A = true)
{
    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x;
    auto const bid_start = hipBlockIdx_z;
    auto const bid_inc = hipGridDim_z;

    auto const even_n = n + (n % 2);
    auto const half_n = even_n / 2;
    auto const ntables = half_n;

    // get dimensions of 2D thread array
    I ddx = NX_THREADS;
    I ddy = RSYEVJ_BDIM / ddx;

    bool const need_V = (esort != rocblas_esort_none);

    // check whether to use A_ and V_ in LDS

    // ----------------------------------------------------------
    // array cosine also used in comuputing matrix Frobenius norm
    // ----------------------------------------------------------
    size_t const size_cosine = sizeof(S) * n;
    size_t const size_sine = sizeof(T) * ntables;

    size_t const size_A = sizeof(T) * n * n;
    size_t const size_V = (need_V) ? sizeof(T) * n * n : 0;

    size_t const total_size = size_cosine + size_sine + size_A + size_V;

    size_t const max_lds = 64 * 1024;
    bool const can_use_lds = (total_size <= max_lds);
    // ---------------------------------------------
    // check cosine and sine arrays still fit in LDS
    // ---------------------------------------------
    assert((size_cosine + size_sine) <= max_lds);

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        // array pointers
        T* const dA = load_ptr_batch<T>(AA, bid, shiftA, strideA);
        T* const Acpy = AcpyA + bid * strideAcpyA;
        T* const dV = Acpy;

        S* const W = WW + bid * strideW;
        S* const residual = residualA + bid;
        I* const n_sweeps = n_sweepsA + bid;
        I* const info = infoA + bid;

        // shared memory
        extern __shared__ double lmem[];

        std::byte* pfree = (std::byte*)&(lmem[0]);

        S* cosines_res = (S*)pfree;
        pfree += size_cosine;
        T* sines_diag = (T*)pfree;
        pfree += size_sine;

        T* A_ = dA;
        T* V_ = (need_V) ? dV : nullptr;

        if(can_use_lds)
        {
            T* A_ = (T*)pfree;
            pfree += size_A;

            if(need_V)
            {
                V_ = (T*)pfree;
                pfree += size_V;
            }
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

        if(need_V && do_overwrite_A)
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
                                           size_t* size_dwork_byte)
{
    // set workspace to zero
    {
        *size_Acpy = 0;
        *size_J = 0;
        *size_dwork_byte = 0;
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
        *size_dwork_byte = sizeof(I) * (n_even * (n_even - 1));
        return;
    }

    auto const ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };
    auto const is_even = [](auto n) { return ((n % 2) == 0); };

    I const n_even = n + (n % 2);
    I const half_n = n_even / 2;

    *size_Acpy = (sizeof(T) * n * n) * batch_count;

    // size of copy of eigen vectors
    if(need_V)
    {
        *size_J = (sizeof(T) * n * n) * batch_count;
    }

    size_t total_bytes = 0;
    bool const is_small_n = (n <= RSYEVJ_BLOCKED_SWITCH(T, need_V));

    if(is_small_n)
    {
        I const nplayers_small = n_even;
        I const len_schedule_small = nplayers_small * (nplayers_small - 1);
        size_t const size_schedule_small = sizeof(I) * len_schedule_small;
        total_bytes += size_schedule_small;
    }
    else
    {
        bool const rsyevj_need_V = true;
        I const nb = get_nb<T>(n, rsyevj_need_V);

        I const nblocks = ceil(n, nb);
        I const nblocks_even = nblocks + (nblocks % 2);
        I const nblocks_half = nblocks_even / 2;
        assert(is_even(nblocks));

        I const nplayers_small = (2 * nb);
        I const nplayers_large = nblocks_even;
        I const len_schedule_small = nplayers_small * (nplayers_small - 1);
        I const len_schedule_large = nplayers_large * (nplayers_large - 1);
        // -----------------------------------------------------
        // other arrays allocated out of a single dwork(:) array
        // -----------------------------------------------------

        size_t const size_merged_blocks_bytes = sizeof(T) * (nb * 2) * (nb * 2);

        size_t const size_Vj_bytes = size_merged_blocks_bytes * (nblocks_half - 1) * batch_count;
        size_t const size_Vj_last_bytes = size_merged_blocks_bytes * batch_count;
        size_t const size_Aj_last_bytes = size_merged_blocks_bytes * batch_count;
        size_t const size_Aj_bytes = size_merged_blocks_bytes * (nblocks_half - 1) * batch_count;

        total_bytes += size_Vj_bytes;
        total_bytes += size_Vj_last_bytes;
        total_bytes += size_Aj_last_bytes;
        total_bytes += size_Aj_bytes;

        size_t const size_Vj_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Aj_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;

        total_bytes += size_Vj_ptr_array + size_Aj_ptr_array;

        size_t const size_Vj_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Aj_last_ptr_array = sizeof(T*) * 1 * batch_count;

        total_bytes += size_Vj_last_ptr_array + size_Aj_last_ptr_array;

        size_t const size_A_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_last_row_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_A_last_col_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_A_ptr_array = sizeof(T*) * batch_count;

        total_bytes += size_A_row_ptr_array + size_A_col_ptr_array + size_A_last_row_ptr_array
            + size_A_last_col_ptr_array + size_A_ptr_array;

        size_t const size_Atmp_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Atmp_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Atmp_last_row_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Atmp_last_col_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Atmp_ptr_array = sizeof(T*) * batch_count;

        total_bytes += size_Atmp_row_ptr_array + size_Atmp_col_ptr_array
            + size_Atmp_last_row_ptr_array + size_Atmp_last_col_ptr_array + size_Atmp_ptr_array;

        size_t const size_Vtmp_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Vtmp_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Vtmp_last_row_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Vtmp_last_col_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Vtmp_ptr_array = sizeof(T) * batch_count;

        total_bytes += size_Vtmp_row_ptr_array + size_Vtmp_col_ptr_array
            + size_Vtmp_last_row_ptr_array + size_Vtmp_last_col_ptr_array + size_Vtmp_ptr_array;

        size_t const size_A_diag_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_last_diag_ptr_array = sizeof(T*) * 1 * batch_count;

        total_bytes += size_A_diag_ptr_array + size_A_last_diag_ptr_array;

        size_t const size_residual_Aj
            = sizeof(S) * (2 * nb) * (2 * nb) * batch_count * (nblocks_half - 1);
        size_t const size_residual_Aj_last = sizeof(S) * (2 * nb) * (2 * nb) * batch_count;
        total_bytes += size_residual_Aj;
        total_bytes += size_residual_Aj_last;

        size_t const size_info_Aj = sizeof(I) * batch_count * (nblocks_half - 1);
        size_t const size_info_Aj_last = sizeof(I) * batch_count;
        total_bytes += size_info_Aj;
        total_bytes += size_info_Aj_last;

        size_t const size_W_Aj = sizeof(S) * (2 * nb) * batch_count * (nblocks_half - 1);
        size_t const size_W_Aj_last = sizeof(S) * (2 * nb) * batch_count;
        total_bytes += size_W_Aj;
        total_bytes += size_W_Aj_last;

        size_t const size_n_sweeps_Aj = sizeof(I) * (nblocks_half - 1) * batch_count;
        size_t const size_n_sweeps_Aj_last = sizeof(I) * batch_count;
        total_bytes += size_n_sweeps_Aj;
        total_bytes += size_n_sweeps_Aj_last;

        size_t const size_work_rocblas = sizeof(T*) * (nblocks_half)*batch_count;
        total_bytes += size_work_rocblas;

        size_t const size_Gmat = sizeof(S) * (nblocks * nblocks) * batch_count;
        total_bytes += size_Gmat;

        size_t const size_schedule_small = sizeof(I) * len_schedule_small;
        size_t const size_schedule_large = sizeof(I) * len_schedule_large;

        total_bytes += size_schedule_small + size_schedule_large;

        *size_dwork_byte = total_bytes;
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
                                                const Istride shiftA,
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
                                                S* norms,
                                                I* completed,
                                                T* dwork,
                                                size_t size_dwork_byte)
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

    bool const need_sort = (esort != rocblas_esort_none);

    Istride const shift_zero = 0;

    auto Atmp = Acpy;
    auto const shiftAtmp = shift_zero;
    auto const ldatmp = n;
    Istride const strideAtmp = ldatmp * n;

    auto Vtmp = J;
    auto const shiftVtmp = shift_zero;
    auto const ldvtmp = n;
    Istride const strideVtmp = ldvtmp * n;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I blocksReset = (batch_count - 1) / BS1 + 1;
    dim3 gridReset(blocksReset, 1, 1);
    dim3 threadsReset(BS1, 1, 1);

    HIP_CHECK(hipMemsetAsync(residual, 0, sizeof(S) * batch_count, stream));
    HIP_CHECK(hipMemsetAsync(n_sweeps, 0, sizeof(I) * batch_count, stream));
    HIP_CHECK(hipMemsetAsync(info, 0, sizeof(I) * batch_count, stream));
    HIP_CHECK(hipMemsetAsync(completed, 0, sizeof(I) * (batch_count + 1), stream));

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
    auto is_odd = [](auto n) { return ((n % 2) != 0); };

    // --------------------------------------------------------------
    // zero out arrays to make sure there are no uninitialized values
    // --------------------------------------------------------------
    {
        int const ivalue = 0;
        HIP_CHECK(hipMemsetAsync((void*)dwork, ivalue, size_dwork_byte, stream));
    }

    std::byte* pfree = (std::byte*)dwork;

    // absolute tolerance for evaluating when the algorithm has converged
    S const eps = get_epsilon<S>();
    S const atol = (abstol <= 0 ? eps : abstol);

    // local variables
    I const even_n = n + (n % 2);
    I const n_even = even_n;
    I const half_n = even_n / 2;

    bool const need_V = (evect != rocblas_evect_none) && (Vtmp != nullptr);

    auto swap = [](auto& x, auto& y) {
        auto const t = x;
        x = y;
        y = t;
    };

    if(n <= RSYEVJ_BLOCKED_SWITCH(T, need_V))
    {
        // *** USE SINGLE SMALL-SIZE KERNEL ***
        // (TODO: RSYEVJ_BLOCKED_SWITCH may need re-tuning
        dim3 grid(1, 1, batch_count);

        I ddx = NX_THREADS;
        I ddy = RSYEVJ_BDIM / ddx;

        I const n_even = n + (n % 2);
        I const nplayers = n_even;
        I const len_schedule_small = nplayers * (nplayers - 1);
        std::vector<I> h_schedule_small(len_schedule_small);

        std::byte* pfree = (std::byte*)dwork;
        I* const d_schedule_small = (I*)pfree;
        size_t const size_schedule_small = sizeof(I) * len_schedule_small;
        pfree += size_schedule_small;

        assert(pfree <= ((std::byte*)dwork) + size_dwork_byte);

        setup_schedule(nplayers, h_schedule_small, d_schedule_small, stream);

        {
            size_t const lmemsize = 64 * 1024;
            Istride const strideAcpy = n * n;
            ROCSOLVER_LAUNCH_KERNEL((rsyevj_small_kernel<T, I, S, U, Istride>),
                                    dim3(1, 1, batch_count), dim3(ddx, ddy, 1), lmemsize, stream,
                                    esort, evect, uplo, n, A, shiftA, lda, strideA, atol, eps,
                                    residual, max_sweeps, n_sweeps, W, strideW, info, Acpy,
                                    strideAcpy, batch_count, d_schedule_small);
        }
    }
    else
    {
        // ------------------------
        // determine block size "nb" and
        // number of blocks "nblocks"
        // ------------------------
        bool const rsyevj_need_V = true;
        auto const nb = get_nb<T>(n, rsyevj_need_V);
        auto const nblocks = ceil(n, nb);
        assert(is_even(nblocks));

        I const even_nblocks = nblocks + (nblocks % 2);
        I const nblocks_half = even_nblocks / 2;

        I const nb_last = n - (nblocks - 1) * nb;
        assert(nb_last >= 1);

        I const nplayers_small = (2 * nb);
        I const nplayers_large = even_nblocks;
        I const len_schedule_small = nplayers_small * (nplayers_small - 1);
        I const len_schedule_large = nplayers_large * (nplayers_large - 1);
        std::vector<I> h_schedule_small(len_schedule_small);
        std::vector<I> h_schedule_large(len_schedule_large);

        auto const num_rounds = (even_nblocks - 1);

        // --------------------------------------
        // preallocate storage for pointer arrays
        // --------------------------------------

        size_t total_bytes = 0;

        size_t const size_merged_blocks_bytes = sizeof(T) * (nb * 2) * (nb * 2);
        // size_t const size_completed = sizeof(I) * (batch_count + 1);
        // total_bytes += size_completed;

        size_t const size_Aj_bytes = size_merged_blocks_bytes * (nblocks_half - 1) * batch_count;
        size_t const size_Vj_bytes = size_merged_blocks_bytes * (nblocks_half - 1) * batch_count;
        size_t const size_Aj_last_bytes = size_merged_blocks_bytes * batch_count;
        size_t const size_Vj_last_bytes = size_merged_blocks_bytes * batch_count;

        total_bytes += size_Aj_bytes;
        total_bytes += size_Vj_bytes;
        total_bytes += size_Aj_last_bytes;
        total_bytes += size_Vj_last_bytes;

        size_t const size_Vj_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Aj_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;

        total_bytes += size_Vj_ptr_array + size_Aj_ptr_array;

        size_t const size_Vj_last_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Aj_last_ptr_array = sizeof(T*) * 1 * batch_count;

        total_bytes += size_Vj_last_ptr_array + size_Aj_last_ptr_array;

        size_t const size_A_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_last_row_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_A_last_col_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_A_ptr_array = sizeof(T*) * batch_count;

        total_bytes += size_A_row_ptr_array + size_A_col_ptr_array + size_A_last_row_ptr_array
            + size_A_last_col_ptr_array + size_A_ptr_array;

        size_t const size_Atmp_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Atmp_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Atmp_last_row_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Atmp_last_col_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Atmp_ptr_array = sizeof(T*) * batch_count;

        total_bytes += size_Atmp_row_ptr_array + size_Atmp_col_ptr_array
            + size_Atmp_last_row_ptr_array + size_Atmp_last_col_ptr_array + size_Atmp_ptr_array;

        size_t const size_Vtmp_row_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Vtmp_col_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_Vtmp_last_row_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Vtmp_last_col_ptr_array = sizeof(T*) * 1 * batch_count;
        size_t const size_Vtmp_ptr_array = sizeof(T) * batch_count;

        total_bytes += size_Vtmp_row_ptr_array + size_Vtmp_col_ptr_array
            + size_Vtmp_last_row_ptr_array + size_Vtmp_last_col_ptr_array + size_Vtmp_ptr_array;

        size_t const size_A_diag_ptr_array = sizeof(T*) * (nblocks_half - 1) * batch_count;
        size_t const size_A_last_diag_ptr_array = sizeof(T*) * 1 * batch_count;

        total_bytes += size_A_diag_ptr_array + size_A_last_diag_ptr_array;

        size_t const size_residual_Aj
            = sizeof(S) * (2 * nb) * (2 * nb) * batch_count * (nblocks_half - 1);
        size_t const size_residual_Aj_last = sizeof(S) * (2 * nb) * (2 * nb) * batch_count;
        total_bytes += size_residual_Aj;
        total_bytes += size_residual_Aj_last;

        size_t const size_info_Aj = sizeof(I) * batch_count * (nblocks_half - 1);
        size_t const size_info_Aj_last = sizeof(I) * batch_count;
        total_bytes += size_info_Aj;
        total_bytes += size_info_Aj_last;

        size_t const size_W_Aj = sizeof(S) * (2 * nb) * batch_count * (nblocks_half - 1);
        size_t const size_W_Aj_last = sizeof(S) * (2 * nb) * batch_count;
        total_bytes += size_W_Aj;
        total_bytes += size_W_Aj_last;

        size_t const size_n_sweeps_Aj = sizeof(I) * (nblocks_half - 1) * batch_count;
        size_t const size_n_sweeps_Aj_last = sizeof(I) * batch_count;
        total_bytes += size_n_sweeps_Aj;
        total_bytes += size_n_sweeps_Aj_last;

        size_t const size_work_rocblas = sizeof(T*) * (nblocks_half)*batch_count;
        total_bytes += size_work_rocblas;

        size_t const size_Gmat = sizeof(S) * (nblocks * nblocks) * batch_count;
        total_bytes += size_Gmat;

        size_t const size_schedule_small = sizeof(I) * len_schedule_small;
        size_t const size_schedule_large = sizeof(I) * len_schedule_large;

        std::byte* pfree = (std::byte*)dwork;

        // I* const completed = (I*)pfree;
        // pfree += size_completed;

        T* const Aj = (T*)pfree;
        pfree += size_Aj_bytes;

        T* const Vj = (T*)pfree;
        pfree += size_Vj_bytes;

        T* const Aj_last = (T*)pfree;
        pfree += size_Aj_last_bytes;

        T* const Vj_last = (T*)pfree;
        pfree += size_Vj_last_bytes;

        I const ldvj = (2 * nb);
        Istride const strideVj = static_cast<Istride>(nblocks_half - 1) * (ldvj * (2 * nb));
        auto const shiftVj = shift_zero;

        I const ldaj = (2 * nb);
        Istride const strideAj = static_cast<Istride>(nblocks_half - 1) * (ldaj * (2 * nb));
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

        T** const A_last_row_ptr_array = (T**)pfree;
        pfree += size_A_last_row_ptr_array;

        T** const A_last_col_ptr_array = (T**)pfree;
        pfree += size_A_last_col_ptr_array;

        T** const A_ptr_array = (T**)pfree;
        pfree += size_A_ptr_array;

        T** Atmp_row_ptr_array = (T**)pfree;
        pfree += size_Atmp_row_ptr_array;

        T** Atmp_col_ptr_array = (T**)pfree;
        pfree += size_Atmp_col_ptr_array;

        T** Atmp_last_row_ptr_array = (T**)pfree;
        pfree += size_Atmp_last_row_ptr_array;

        T** Atmp_last_col_ptr_array = (T**)pfree;
        pfree += size_Atmp_last_col_ptr_array;

        T** Atmp_ptr_array = (T**)pfree;
        pfree += size_Atmp_ptr_array;

        T** Vtmp_row_ptr_array = (T**)pfree;
        pfree += size_Vtmp_row_ptr_array;

        T** Vtmp_col_ptr_array = (T**)pfree;
        pfree += size_Vtmp_col_ptr_array;

        T** Vtmp_last_row_ptr_array = (T**)pfree;
        pfree += size_Vtmp_last_row_ptr_array;

        T** Vtmp_last_col_ptr_array = (T**)pfree;
        pfree += size_Vtmp_last_col_ptr_array;

        T** Vtmp_ptr_array = (T**)pfree;
        pfree += size_Vtmp_ptr_array;

        T** const A_diag_ptr_array = (T**)pfree;
        pfree += size_A_diag_ptr_array;

        T** const A_last_diag_ptr_array = (T**)pfree;
        pfree += size_A_last_diag_ptr_array;

        S* const residual_Aj = (S*)pfree;
        pfree += size_residual_Aj;

        S* const residual_Aj_last = (S*)pfree;
        pfree += size_residual_Aj_last;

        I* const info_Aj = (I*)pfree;
        pfree += size_info_Aj;

        I* const info_Aj_last = (I*)pfree;
        pfree += size_info_Aj_last;

        S* const W_Aj = (S*)pfree;
        pfree += size_W_Aj;

        S* const W_Aj_last = (S*)pfree;
        pfree += size_W_Aj_last;

        I* const n_sweeps_Aj = (I*)pfree;
        pfree += size_n_sweeps_Aj;

        I* const n_sweeps_Aj_last = (I*)pfree;
        pfree += size_n_sweeps_Aj_last;

        S* const Gmat = (S*)pfree;
        pfree += size_Gmat;

        S* const Amat_norm = norms;

        I* const d_schedule_small = (I*)pfree;
        pfree += size_schedule_small;

        I* const d_schedule_large = (I*)pfree;
        pfree += size_schedule_large;

        T** const work_rocblas = (T**)pfree;
        pfree += size_work_rocblas;

        assert(pfree <= (((std::byte*)dwork) + size_dwork_byte));

        bool constexpr use_adjust_schedule = true;

        {
            setup_schedule(nplayers_small, h_schedule_small, d_schedule_small, stream);

            setup_schedule(nplayers_large, h_schedule_large, d_schedule_large, stream,
                           use_adjust_schedule);
        }

        char const c_uplo = (uplo == rocblas_fill_upper) ? 'U'
            : (uplo == rocblas_fill_lower)               ? 'L'
                                                         : 'A';
        // ---------------------
        // launch configurations
        // ---------------------
        auto const max_lds = 64 * 1024;
        auto const max_blocks = 64 * 1000;
        auto const nx = NX_THREADS;
        auto const ny = RSYEVJ_BDIM / nx;

        auto const nbx = std::min(max_blocks, ceil(n, nx));
        auto const nby = std::min(max_blocks, ceil(n, ny));
        auto const nbz = std::min(max_blocks, batch_count);
        {
            // ---------------------------
            // make matrix to be symmetric
            // ---------------------------

            ROCSOLVER_LAUNCH_KERNEL((symmetrize_matrix_kernel<T, I, U, Istride>),
                                    dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream, c_uplo, n, A,
                                    shiftA, lda, strideA, batch_count);
        }

        // ----------------------------------
        // precompute norms of orginal matrix
        // ----------------------------------
        {
            bool const include_diagonal = true;
            ROCSOLVER_LAUNCH_KERNEL((cal_Gmat_kernel<T, I, S, Istride, U>), dim3(nbx, nby, nbz),
                                    dim3(nx, ny, 1), 0, stream, n, nb, A, shiftA, lda, strideA,
                                    Gmat, include_diagonal, completed, batch_count);

            auto const shmem_size = sizeof(S);
            ROCSOLVER_LAUNCH_KERNEL((sum_Gmat<S, I>), dim3(1, 1, nbz), dim3(nx, ny, 1), shmem_size,
                                    stream, n, nb, Gmat, residual, completed, batch_count);
        }

        I n_completed = 0;
        I h_sweeps = 0;
        bool is_converged = false;

        for(h_sweeps = 0; h_sweeps < max_sweeps; h_sweeps++)
        {
            {
                // compute norms of off diagonal blocks
                // setup completed[] array

                // -----------------------------------------------------
                // compute norms of blocks into array
                //
                // Gmat(0:(nblocks-1), 0:(nblocks-1), 0:(batch_count-1))
                // -----------------------------------------------------

                bool const need_diagonal = false;
                ROCSOLVER_LAUNCH_KERNEL((cal_Gmat_kernel<T, I, S, Istride, U>), dim3(nbx, nby, nbz),
                                        dim3(nx, ny, 1), 0, stream,

                                        n, nb, A, shiftA, lda, strideA, Gmat, need_diagonal,
                                        completed, batch_count);

                size_t const shmem_size = sizeof(S);
                ROCSOLVER_LAUNCH_KERNEL((sum_Gmat<S, I>), dim3(1, 1, nbz), dim3(nx, ny, 1),
                                        shmem_size, stream, n, nb, Gmat, residual, completed,
                                        batch_count);

                {
                    // --------------------------------------------
                    // zero out just complete[0] to count number of
                    // completed batch entries
                    // --------------------------------------------
                    int const ivalue = 0;
                    size_t const nbytes = sizeof(I);
                    HIP_CHECK(hipMemsetAsync(&(completed[0]), ivalue, nbytes, stream));
                }

                auto const nnx = 64;
                auto const nnb = ceil(batch_count, nnx);
                ROCSOLVER_LAUNCH_KERNEL((set_completed_kernel<S, I, Istride>), dim3(nnb, 1, 1),
                                        dim3(nnx, 1, 1), 0, stream, n, nb, Amat_norm, abstol,
                                        h_sweeps, n_sweeps, residual, info, completed, batch_count);
            }

            {
                // --------------------------------------
                // check convergence of all batch entries
                // --------------------------------------
                void* dst = (void*)&(n_completed);
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
            assert((1 <= batch_count_remain) && (batch_count_remain <= batch_count));

            {
                // ------------------------------------------
                // build pointer arrays for data movement and
                // for rocblas batch GEMM operations
                // ------------------------------------------

                // ---------------------------------
                // reset value to be used as counter
                // ---------------------------------
                HIP_CHECK(hipMemsetAsync((void*)&(completed[0]), 0, sizeof(I), stream));

                auto const nx = NX_THREADS;
                ROCSOLVER_LAUNCH_KERNEL(
                    (setup_ptr_arrays_kernel<T, I, Istride, U, T*, T*>), dim3(1, 1, nbz),
                    dim3(nx, 1, 1), 0, stream,

                    n, nb,

                    A, shiftA, lda, strideA,

                    Atmp, shiftAtmp, ldatmp, strideAtmp,

                    Vtmp, shiftVtmp, ldvtmp, strideVtmp,

                    Aj, strideAj, Vj, strideVj, Aj_last, Vj_last, completed,

                    Vj_ptr_array, Aj_ptr_array, Vj_last_ptr_array, Aj_last_ptr_array,

                    A_row_ptr_array, A_col_ptr_array, A_last_row_ptr_array, A_last_col_ptr_array,
                    A_ptr_array,

                    Atmp_row_ptr_array, Atmp_col_ptr_array, Atmp_last_row_ptr_array,
                    Atmp_last_col_ptr_array, Atmp_ptr_array,

                    Vtmp_row_ptr_array, Vtmp_col_ptr_array, Vtmp_last_row_ptr_array,
                    Vtmp_last_col_ptr_array, Vtmp_ptr_array,

                    A_diag_ptr_array, A_last_diag_ptr_array, batch_count);

#ifdef NDEBUG
#else

                check_ptr_array("Vj_ptr_array", (nblocks_half - 1) * batch_count_remain,
                                Vj_ptr_array);
                check_ptr_array("Vj_last_ptr_array", batch_count_remain, Vj_last_ptr_array);

                check_ptr_array("A_row_ptr_array", (nblocks_half - 1) * batch_count_remain,
                                A_row_ptr_array);
                check_ptr_array("A_col_ptr_array", (nblocks_half - 1) * batch_count_remain,
                                A_col_ptr_array);
                check_ptr_array("A_last_row_ptr_array", batch_count_remain, A_last_row_ptr_array);
                check_ptr_array("A_last_col_ptr_array", batch_count_remain, A_last_col_ptr_array);
                check_ptr_array("A_diag_ptr_array", (nblocks_half - 1) * batch_count_remain,
                                A_diag_ptr_array);
                check_ptr_array("A_last_diag_ptr_array", batch_count_remain, A_last_diag_ptr_array);

                check_ptr_array("A_ptr_array", batch_count, A_ptr_array);
                check_ptr_array("Aj_last_ptr_array", batch_count_remain, Aj_last_ptr_array);

                check_ptr_array("Atmp_row_ptr_array", (nblocks_half - 1) * batch_count_remain,
                                Atmp_row_ptr_array);
                check_ptr_array("Atmp_col_ptr_array", (nblocks_half - 1) * batch_count_remain,
                                Atmp_col_ptr_array);
                check_ptr_array("Atmp_last_row_ptr_array", batch_count_remain,
                                Atmp_last_row_ptr_array);
                check_ptr_array("Atmp_last_col_ptr_array", batch_count_remain,
                                Atmp_last_col_ptr_array);
                check_ptr_array("Atmp_ptr_array", batch_count, Atmp_ptr_array);

                if(need_V)
                {
                    check_ptr_array("Vtmp_ptr_array", batch_count, Vtmp_ptr_array);
                    check_ptr_array("Vtmp_row_ptr_array", (nblocks_half - 1) * batch_count_remain,
                                    Vtmp_row_ptr_array);
                    check_ptr_array("Vtmp_col_ptr_array", (nblocks_half - 1) * batch_count_remain,
                                    Vtmp_col_ptr_array);
                    check_ptr_array("Vtmp_last_row_ptr_array", batch_count_remain,
                                    Vtmp_last_row_ptr_array);
                    check_ptr_array("Vtmp_last_col_ptr_array", batch_count_remain,
                                    Vtmp_last_col_ptr_array);
                }

#endif
            }

            for(I iround = 0; iround < num_rounds; iround++)
            {
                // ------------------------
                // reorder and copy to Atmp, Vtmp
                // ------------------------
                I const* const col_map = d_schedule_large + iround * (even_nblocks);
                I const* const row_map = col_map;

                auto const max_blocks = 64 * 1000;
                auto const nx = NX_THREADS;
                auto const ny = RSYEVJ_BDIM / nx;

                auto const nbx = std::min(max_blocks, ceil(n, nx));
                auto const nby = std::min(max_blocks, ceil(n, ny));
                auto const nbz = std::min(max_blocks, batch_count_remain);

                {
                    char const c_direction = 'F'; // forward direction

                    if(need_V)
                    {
                        // ------------------------------------
                        // matrix V need only column reordering
                        // ------------------------------------
                        I const* const null_row_map = nullptr;

                        ROCSOLVER_LAUNCH_KERNEL((reorder_kernel<T, I, Istride>),
                                                dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream,
                                                c_direction, n, nb, null_row_map, col_map, Vtmp,
                                                shiftVtmp, ldvtmp, strideVtmp, Atmp, shiftAtmp,
                                                ldatmp, strideAtmp, batch_count_remain);

                        swap(Atmp, Vtmp);
                        swap(Atmp_row_ptr_array, Vtmp_row_ptr_array);
                        swap(Atmp_col_ptr_array, Vtmp_col_ptr_array);
                        swap(Atmp_last_row_ptr_array, Vtmp_last_row_ptr_array);
                        swap(Atmp_last_col_ptr_array, Vtmp_last_col_ptr_array);
                        swap(Atmp_ptr_array, Vtmp_ptr_array);
                    }

                    ROCSOLVER_LAUNCH_KERNEL((reorder_kernel<T, I, Istride>), dim3(nbx, nby, nbz),
                                            dim3(nx, ny, 1), 0, stream, c_direction, n, nb, row_map,
                                            col_map, A, shiftA, lda, strideA, Atmp, shiftAtmp,
                                            ldatmp, strideAtmp, batch_count_remain);
                }

                // ------------------------------------------------------
                // prepare to perform Jacobi iteration on independent sets of blocks
                // ------------------------------------------------------

                {

                    {// --------------------------
                     // copy diagonal blocks to Aj
                     // --------------------------

                     I const m1 = (2 * nb);
                I const n1 = (2 * nb);
                char const cl_uplo = 'A';

                Istride const strideA_diag = 0;
                I const lbatch_count = (nblocks_half - 1) * batch_count_remain;

                ROCSOLVER_LAUNCH_KERNEL((lacpy_kernel<T, I, T**, T**, Istride>),
                                        dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream, cl_uplo,
                                        m1, n1, A_diag_ptr_array, shift_zero, lda, strideA_diag,
                                        Aj_ptr_array, shiftAj, ldaj, strideAj, lbatch_count);

                I const m2 = (nb + nb_last);
                I const n2 = (nb + nb_last);
                ROCSOLVER_LAUNCH_KERNEL(
                    (lacpy_kernel<T, I, T**, T**, Istride>), dim3(nbx, nby, nbz), dim3(nx, ny, 1),
                    0, stream, cl_uplo, m2, n2, A_last_diag_ptr_array, shiftA, lda, strideA,
                    Aj_last_ptr_array, shiftAj, ldaj, strideAj, batch_count_remain);
            }

            {
                // -------------------------------------------------------
                // prepare to perform Jacobi iteration on small diagonal blocks in Aj
                // -------------------------------------------------------

                // ----------------------------
                // set Vj to be diagonal matrix
                // to store the matrix of eigen vectors
                // ----------------------------

                {
                    char const c_uplo = 'A';
                    I const mm = (2 * nb);
                    I const nn = (2 * nb);

                    T alpha_offdiag = 0;
                    T beta_diag = 1;

                    ROCSOLVER_LAUNCH_KERNEL((laset_kernel<T, I, T*, Istride>), dim3(nbx, nby, nbz),
                                            dim3(nx, ny, 1), 0, stream, c_uplo, mm, nn,
                                            alpha_offdiag, beta_diag, Vj, shiftVj, ldvj, strideVj,
                                            (nblocks_half - 1) * batch_count_remain);

                    ROCSOLVER_LAUNCH_KERNEL((laset_kernel<T, I, T*, Istride>), dim3(nbx, nby, nbz),
                                            dim3(nx, ny, 1), 0, stream, c_uplo, mm, nn,
                                            alpha_offdiag, beta_diag, Vj_last, shiftVj, ldvj,
                                            strideVj, batch_count_remain);
                }

                {
                    // -----------------------------------------
                    // setup options to
                    // preserve the nearly diagonal matrix in Aj
                    // -----------------------------------------
                    bool const do_overwrite_A = false;
                    rocblas_esort const rsyevj_esort = rocblas_esort_none;
                    size_t const lmemsize = 64 * 1024;

                    // ---------------------------
                    // no need for too many sweeps
                    // since the blocks will be over-written
                    // ---------------------------
                    I const rsyevj_max_sweeps = 15;
                    auto const rsyevj_atol = atol / nblocks;

                    // -----------------------------------------
                    // need to store the matrix of eigen vectors
                    // -----------------------------------------
                    rocblas_evect const rsyevj_evect = rocblas_evect_original;
                    TRACE(2);
                    // TODO: need Vj( (2*nb), (2*nb), (nblock/2-1), batch_count)
                    // and Vj_last( (2*nb), (2*nb), batch_count )
                    //
                    I const n1 = (2 * nb);
                    Istride const lstride_Vj = (2 * nb) * (2 * nb);
                    Istride const strideW_Aj = (2 * nb);
                    ROCSOLVER_LAUNCH_KERNEL(
                        (rsyevj_small_kernel<T, I, S, T**, Istride>), dim3(1, 1, nbz),
                        dim3(nx, ny, 1), lmemsize, stream, rsyevj_esort, rsyevj_evect, uplo, n1,
                        Aj_ptr_array, shiftAj, ldaj, strideAj, rsyevj_atol, eps, residual_Aj,
                        rsyevj_max_sweeps, n_sweeps_Aj, W_Aj, strideW_Aj, info_Aj, Vj, lstride_Vj,
                        (nblocks_half - 1) * batch_count_remain, d_schedule_small, do_overwrite_A);
                    TRACE(2);

                    I const n2 = nb + nb_last;
                    Istride const lstride_Vj_last = (2 * nb) * (2 * nb);
                    Istride const strideW_Aj_last = (2 * nb);
                    ROCSOLVER_LAUNCH_KERNEL(
                        (rsyevj_small_kernel<T, I, S, T**, Istride>), dim3(1, 1, nbz),
                        dim3(nx, ny, 1), lmemsize, stream, rsyevj_esort, rsyevj_evect, uplo, n2,
                        Aj_last_ptr_array, shiftAj, ldaj, strideAj, rsyevj_atol, eps,
                        residual_Aj_last, rsyevj_max_sweeps, n_sweeps_Aj_last, W_Aj_last,
                        strideW_Aj_last, info_Aj_last, Vj_last, lstride_Vj_last, batch_count_remain,
                        d_schedule_small, do_overwrite_A);
                    TRACE(2);
                }
            }
        }

        {
            // -----------------------------------------------------
            // launch batch list to perform Vj' to update block rows
            // -----------------------------------------------------

            T alpha = 1;
            T beta = 0;
            auto m1 = 2 * nb;
            auto n1 = n;
            auto k1 = 2 * nb;

            rocblas_operation const transA = rocblas_operation_conjugate_transpose;
            rocblas_operation const transB = rocblas_operation_none;

            auto const lbatch_count = (nblocks_half - 1) * batch_count_remain;
            TRACE(2);
            ROCBLAS_CHECK(rocblasCall_gemm(handle, transA, transB, m1, n1, k1, &alpha, Vj,
                                           shift_zero, ldvj, strideVj, Atmp_row_ptr_array,
                                           shift_zero, ldatmp, strideAtmp, &beta, A_row_ptr_array,
                                           shift_zero, lda, strideA, lbatch_count, work_rocblas));
            TRACE(2);
            // ----------------------------------------------------------
            // launch batch list to perform Vj' to update last block rows
            // ----------------------------------------------------------
            auto m2 = n;
            auto n2 = nb + nb_last;
            auto k2 = nb + nb_last;

            ROCBLAS_CHECK(rocblasCall_gemm(
                handle, transA, transB, m2, n2, k2, &alpha, Vj, shift_zero, ldvj, strideVj,
                Atmp_last_row_ptr_array, shift_zero, ldatmp, strideAtmp, &beta,
                A_last_row_ptr_array, shift_zero, lda, strideA, batch_count_remain, work_rocblas));
            TRACE(2);
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

            rocblas_operation const transA = rocblas_operation_none;
            rocblas_operation const transB = rocblas_operation_none;

            auto const lbatch_count = (nblocks_half - 1) * batch_count_remain;

            TRACE(2);
            ROCBLAS_CHECK(rocblasCall_gemm(handle, transA, transB, m1, n1, k1, &alpha, Vj_ptr_array,
                                           shift_zero, ldvj, strideVj, A_col_ptr_array, shift_zero,
                                           lda, strideA, &beta, Atmp_col_ptr_array, shift_zero,
                                           ldatmp, strideAtmp, lbatch_count, work_rocblas));

            // -----------------------------------------------------------
            // launch batch list to perform Vj to update last block column
            // -----------------------------------------------------------

            I m2 = n;
            I n2 = nb_last;
            I k2 = nb_last;

            TRACE(2);
            ROCBLAS_CHECK(rocblasCall_gemm(handle, transA, transB, m2, n2, k2, &alpha,
                                           Vj_last_ptr_array, shift_zero, ldvtmp, strideVtmp,
                                           A_last_col_ptr_array, shift_zero, lda, strideA, &beta,
                                           Atmp_last_col_ptr_array, shift_zero, ldatmp, strideAtmp,
                                           batch_count_remain, work_rocblas));
            TRACE(2);
            {
                // -------------------
                // copy Atmp back to A
                // and undo reordering while copying
                // -----------------------------
                char const c_direction = 'B';

                ROCSOLVER_LAUNCH_KERNEL(
                    (reorder_kernel<T, I, Istride>), dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0,
                    stream, c_direction, n, nb, (use_adjust_schedule) ? nullptr : row_map,
                    (use_adjust_schedule) ? nullptr : col_map,

                    Atmp, shiftAtmp, ldatmp, strideAtmp, A, shiftA, lda, strideA, batch_count);
            }

            TRACE(2);
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

            ROCBLAS_CHECK(rocblasCall_gemm(
                handle, rocblas_operation_none, rocblas_operation_none, m1, n1, k1, &alpha,
                Vj_ptr_array, shift_zero, ldvj, strideVj, Vtmp_col_ptr_array, shift_zero, ldvtmp,
                strideVtmp, &beta, Atmp_col_ptr_array, shift_zero, ldatmp, strideAtmp,
                (nblocks_half - 1) * batch_count, work_rocblas));

            I m2 = n;
            I n2 = nb + nb_last;
            I k2 = nb + nb_last;

            ROCBLAS_CHECK(rocblasCall_gemm(handle, rocblas_operation_none, rocblas_operation_none,
                                           m2, n2, k2, &alpha, Vj_last_ptr_array, shift_zero, ldvj,
                                           strideVj, Vtmp_last_col_ptr_array, shift_zero, ldvtmp,
                                           strideVtmp, &beta, Atmp_last_col_ptr_array, shift_zero,
                                           ldatmp, strideAtmp, batch_count, work_rocblas));

            swap(Atmp, Vtmp);
            swap(Atmp_row_ptr_array, Vtmp_row_ptr_array);
            swap(Atmp_col_ptr_array, Vtmp_col_ptr_array);
            swap(Atmp_last_row_ptr_array, Vtmp_last_row_ptr_array);
            swap(Atmp_last_col_ptr_array, Vtmp_last_col_ptr_array);
            swap(Atmp_ptr_array, Vtmp_ptr_array);
        }

    } // end for iround

} // end for sweeps

TRACE(2);
{
    // ---------------------
    // copy out eigen values
    // ---------------------

    ROCSOLVER_LAUNCH_KERNEL((copy_diagonal_kernel<T, I, U, Istride>), dim3(1, 1, nbz), dim3(nx, 1, 1),
                            0, stream, n, A, shiftA, lda, strideA, W, strideW, batch_count);
}

TRACE(2);
{
    // -------------------------------------------
    // check whether eigenvalues need to be sorted
    // -------------------------------------------

    // reuse storage
    I* const map = (need_V) ? (I*)Acpy : nullptr;
    Istride const stridemap = sizeof(I) * n;
    if(need_sort)
    {
        ROCSOLVER_LAUNCH_KERNEL((sort_kernel<S, I, Istride>), dim3(1, 1, nbz), dim3(nx, 1, 1), 0,
                                stream,

                                n, W, strideW, map, stridemap, batch_count);
    }
    TRACE(2);

    // -----------------------------------------------
    // over-write original matrix A with eigen vectors
    // -----------------------------------------------
    if(need_V)
    {
        I* const row_map = nullptr;
        I* const col_map = (need_sort) ? map : nullptr;
        auto const mm = n;
        auto const nn = n;
        ROCSOLVER_LAUNCH_KERNEL((gather2D_kernel<T, I, Istride, T*, U>), dim3(nbx, nby, nbz),
                                dim3(nx, ny, 1), 0, stream, mm, nn, row_map, col_map, Vtmp,
                                shiftVtmp, ldvtmp, strideVtmp, A, shiftA, lda, strideA, batch_count);
    }
}
TRACE(2);

{
    // ------------------------------------------------------
    // final evaluation of residuals and test for convergence
    // ------------------------------------------------------

    {
        int ivalue = 0;
        size_t nbytes = sizeof(I) * (batch_count + 1);
        HIP_CHECK(hipMemsetAsync((void*)completed, ivalue, nbytes, stream));
    }

    TRACE(2);
    bool const need_diagonal = false;
    ROCSOLVER_LAUNCH_KERNEL(
        (cal_Gmat_kernel<T, I, S, Istride, U>), dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream,

        n, nb, A, shiftA, lda, strideA, Gmat, need_diagonal, completed, batch_count);

    TRACE(2);

    ROCSOLVER_LAUNCH_KERNEL((sum_Gmat<S, I>), dim3(1, 1, nbz), dim3(nx, ny, 1), sizeof(S), stream,
                            n, nb, Gmat, residual, completed, batch_count);
    auto const nnx = 64;
    auto const nnb = ceil(batch_count, nnx);
    TRACE(2);
    ROCSOLVER_LAUNCH_KERNEL((set_completed_kernel<S, I, Istride>), dim3(nnb, 1, 1), dim3(nnx, 1, 1),
                            0, stream, n, nb, Amat_norm, abstol, h_sweeps, n_sweeps, residual, info,
                            completed, batch_count);
}
TRACE(2);
#ifdef NDEBUG
#else
if(idebug >= 1)
{
    // debug
    std::vector<S> h_Gmat(nblocks * nblocks * batch_count);
    std::vector<I> h_info(batch_count);
    std::vector<S> h_residual(batch_count);
    std::vector<I> h_completed(batch_count + 1);

    HIP_CHECK(hipMemcpy(&(h_Gmat[0]), Gmat, sizeof(S) * nblocks * nblocks * batch_count,
                        hipMemcpyDeviceToHost));
    for(I bid = 0; bid < batch_count; bid++)
    {
        for(I jb = 0; jb < nblocks; jb++)
        {
            for(I ib = 0; ib < nblocks; ib++)
            {
                auto const ij = ib + jb * nblocks + bid * (nblocks * nblocks);
                auto const gij = h_Gmat[ij];
                printf("Gmat(%d,%d,%d) = %le\n", ib, jb, bid, gij);
            }
        }
    }

    HIP_CHECK(hipMemcpy(&(h_info[0]), info, sizeof(I) * batch_count, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&(h_residual[0]), residual, sizeof(S) * batch_count, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&(h_completed[0]), completed, sizeof(I) * (batch_count + 1),
                        hipMemcpyDeviceToHost));

    printf("completed[0] = %d\n", (int)h_completed[0]);
    for(I bid = 0; bid < batch_count; bid++)
    {
        printf("info[%d] = %d, residual[%d] = %le, completed[%d] = %d\n", bid, (int)h_info[bid],
               bid, (double)h_residual[bid], bid, (int)h_completed[bid + 1]);
    }
}
#endif

} // end large block

return (rocblas_status_success);
}

ROCSOLVER_END_NAMESPACE
