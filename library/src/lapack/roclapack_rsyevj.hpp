/************************************************************************
 * Derived from
 * Golub & Van Loan (1996). Matrix Computations (3rd ed.).
 *     John Hopkins University Press.
 *     Section 8.4.
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
        auto const istat = hipDeviceSynchronize();        \
        if(idebug >= ival)                                \
        {                                                 \
            printf("trace: %s:%d\n", __FILE__, __LINE__); \
            fflush(stdout);                               \
        }                                                 \
    }

#define ALLOC_INIT()                                              \
    /* ------------------------                                 \
         determine block size "nb" and                          \
         number of blocks "nblocks"                             \
         ------------------------*/ \
    bool const rsyevj_need_V = true;                              \
    auto const nb = get_nb<T>(n, rsyevj_need_V);                  \
    auto const nblocks = ceil(n, nb);                             \
    assert(is_even(nblocks));                                     \
                                                                  \
    I const even_nblocks = nblocks + (nblocks % 2);               \
    I const nblocks_half = even_nblocks / 2;                      \
                                                                  \
    I const nb_last = n - (nblocks - 1) * nb;                     \
    assert((1 <= nb_last) && (nb_last <= nb));                    \
                                                                  \
    auto const num_rounds = (even_nblocks - 1);                   \
                                                                  \
    auto const lbatch_count = (nblocks_half - 1) * batch_count;   \
                                                                  \
    size_t const size_merged_blocks_bytes = sizeof(T) * (nb * 2) * (nb * 2);

#define ALLOC_VJ()                                                                  \
    I ldvj = (2 * nb);                                                              \
    Istride shift_Vj = 0;                                                           \
    Istride stride_Vj = static_cast<Istride>(nblocks_half - 1) * (ldvj * (2 * nb)); \
    Istride lstride_Vj = ldvj * (2 * nb);                                           \
    Istride shiftVj = 0;                                                            \
                                                                                    \
    size_t const size_Vj_bytes = size_merged_blocks_bytes * lbatch_count;           \
                                                                                    \
    I ldvj_last = (nb + nb_last);                                                   \
    Istride lstride_Vj_last = ldvj_last * (nb + nb_last);                           \
    Istride shift_Vj_last = 0;                                                      \
                                                                                    \
    size_t const size_Vj_last_bytes = sizeof(T) * lstride_Vj_last * batch_count;    \
                                                                                    \
    T* Vj = (T*)pfree;                                                              \
    pfree += size_Vj_bytes;                                                         \
                                                                                    \
    T* Vj_last = (T*)pfree;                                                         \
    pfree += size_Vj_last_bytes;                                                    \
                                                                                    \
    total_bytes += size_Vj_bytes;                                                   \
    total_bytes += size_Vj_last_bytes;                                              \
                                                                                    \
    size_t const size_Vj_ptr_array = sizeof(T*) * lbatch_count;                     \
    size_t const size_Vj_last_ptr_array = sizeof(T*) * 1 * batch_count;             \
                                                                                    \
    T** Vj_ptr_array = (T**)pfree;                                                  \
    pfree += size_Vj_ptr_array;                                                     \
                                                                                    \
    T** Vj_last_ptr_array = (T**)pfree;                                             \
    pfree += size_Vj_last_ptr_array;                                                \
                                                                                    \
    total_bytes += size_Vj_ptr_array;                                               \
    total_bytes += size_Vj_last_ptr_array;

#define ALLOC_AJ()                                                                  \
                                                                                    \
    I ldaj = (2 * nb);                                                              \
    Istride shift_Aj = 0;                                                           \
    Istride stride_Aj = static_cast<Istride>(nblocks_half - 1) * (ldaj * (2 * nb)); \
    Istride lstride_Aj = ldaj * (2 * nb);                                           \
    Istride shiftAj = 0;                                                            \
                                                                                    \
    size_t const size_Aj_bytes = size_merged_blocks_bytes * lbatch_count;           \
                                                                                    \
    I ldaj_last = (nb + nb_last);                                                   \
    Istride shift_Aj_last = 0;                                                      \
    Istride lstride_Aj_last = ldaj_last * (nb + nb_last);                           \
                                                                                    \
    size_t const size_Aj_last_bytes = sizeof(T) * lstride_Aj_last * batch_count;    \
                                                                                    \
    T* Aj = (T*)pfree;                                                              \
    pfree += size_Aj_bytes;                                                         \
                                                                                    \
    T* Aj_last = (T*)pfree;                                                         \
    pfree += size_Aj_last_bytes;                                                    \
                                                                                    \
    total_bytes += size_Aj_bytes;                                                   \
    total_bytes += size_Aj_last_bytes;                                              \
                                                                                    \
    size_t const size_Aj_ptr_array = sizeof(T*) * lbatch_count;                     \
    size_t const size_Aj_last_ptr_array = sizeof(T*) * 1 * batch_count;             \
                                                                                    \
    T** Aj_ptr_array = (T**)pfree;                                                  \
    pfree += size_Aj_ptr_array;                                                     \
                                                                                    \
    T** Aj_last_ptr_array = (T**)pfree;                                             \
    pfree += size_Aj_last_ptr_array;                                                \
                                                                                    \
    total_bytes += size_Aj_ptr_array;                                               \
    total_bytes += size_Aj_last_ptr_array;

#define ALLOC_Atmp()                                                    \
    I ldatmp = n;                                                       \
    Istride shift_Atmp = 0;                                             \
    Istride lstride_Atmp = ldatmp * n;                                  \
                                                                        \
    size_t size_Atmp_row_ptr_array = sizeof(T*) * lbatch_count;         \
    size_t size_Atmp_col_ptr_array = sizeof(T*) * lbatch_count;         \
    size_t size_Atmp_last_row_ptr_array = sizeof(T*) * 1 * batch_count; \
    size_t size_Atmp_last_col_ptr_array = sizeof(T*) * 1 * batch_count; \
    size_t size_Atmp_ptr_array = sizeof(T*) * batch_count;              \
                                                                        \
    T** Atmp_row_ptr_array = (T**)pfree;                                \
    pfree += size_Atmp_row_ptr_array;                                   \
                                                                        \
    T** Atmp_col_ptr_array = (T**)pfree;                                \
    pfree += size_Atmp_col_ptr_array;                                   \
                                                                        \
    T** Atmp_last_row_ptr_array = (T**)pfree;                           \
    pfree += size_Atmp_last_row_ptr_array;                              \
                                                                        \
    T** Atmp_last_col_ptr_array = (T**)pfree;                           \
    pfree += size_Atmp_last_col_ptr_array;                              \
                                                                        \
    T** Atmp_ptr_array = (T**)pfree;                                    \
    pfree += size_Atmp_ptr_array;                                       \
                                                                        \
    total_bytes += size_Atmp_row_ptr_array;                             \
    total_bytes += size_Atmp_col_ptr_array;                             \
    total_bytes += size_Atmp_last_row_ptr_array;                        \
    total_bytes += size_Atmp_last_col_ptr_array;                        \
    total_bytes += size_Atmp_ptr_array;                                 \
                                                                        \
    size_t size_Atmp_diag_ptr_array = sizeof(T*) * lbatch_count;        \
    size_t size_Atmp_last_diag_ptr_array = sizeof(T*) * batch_count;    \
                                                                        \
    T** Atmp_diag_ptr_array = (T**)pfree;                               \
    pfree += size_Atmp_diag_ptr_array;                                  \
                                                                        \
    T** Atmp_last_diag_ptr_array = (T**)pfree;                          \
    pfree += size_Atmp_last_diag_ptr_array;                             \
                                                                        \
    total_bytes += size_Atmp_diag_ptr_array;                            \
    total_bytes += size_Atmp_last_diag_ptr_array;

#define ALLOC_Vtmp()                                                    \
    I ldvtmp = n;                                                       \
    Istride shift_Vtmp = 0;                                             \
    Istride lstride_Vtmp = ldvtmp * n;                                  \
                                                                        \
    size_t size_Vtmp_row_ptr_array = sizeof(T*) * lbatch_count;         \
    size_t size_Vtmp_col_ptr_array = sizeof(T*) * lbatch_count;         \
    size_t size_Vtmp_last_row_ptr_array = sizeof(T*) * 1 * batch_count; \
    size_t size_Vtmp_last_col_ptr_array = sizeof(T*) * 1 * batch_count; \
    size_t size_Vtmp_ptr_array = sizeof(T*) * batch_count;              \
                                                                        \
    T** Vtmp_row_ptr_array = (T**)pfree;                                \
    pfree += size_Vtmp_row_ptr_array;                                   \
                                                                        \
    T** Vtmp_col_ptr_array = (T**)pfree;                                \
    pfree += size_Vtmp_col_ptr_array;                                   \
                                                                        \
    T** Vtmp_last_row_ptr_array = (T**)pfree;                           \
    pfree += size_Vtmp_last_row_ptr_array;                              \
                                                                        \
    T** Vtmp_last_col_ptr_array = (T**)pfree;                           \
    pfree += size_Vtmp_last_col_ptr_array;                              \
                                                                        \
    T** Vtmp_ptr_array = (T**)pfree;                                    \
    pfree += size_Vtmp_ptr_array;                                       \
                                                                        \
    total_bytes += size_Vtmp_row_ptr_array;                             \
    total_bytes += size_Vtmp_col_ptr_array;                             \
    total_bytes += size_Vtmp_last_row_ptr_array;                        \
    total_bytes += size_Vtmp_last_col_ptr_array;                        \
    total_bytes += size_Vtmp_ptr_array;                                 \
                                                                        \
    size_t size_Vtmp_diag_ptr_array = sizeof(T*) * lbatch_count;        \
    size_t size_Vtmp_last_diag_ptr_array = sizeof(T*) * batch_count;    \
                                                                        \
    T** Vtmp_diag_ptr_array = (T**)pfree;                               \
    pfree += size_Vtmp_diag_ptr_array;                                  \
                                                                        \
    T** Vtmp_last_diag_ptr_array = (T**)pfree;                          \
    pfree += size_Vtmp_last_diag_ptr_array;                             \
                                                                        \
    total_bytes += size_Vtmp_diag_ptr_array;                            \
    total_bytes += size_Vtmp_last_diag_ptr_array;

#define ALLOC_A()                                                          \
    size_t const size_A_row_ptr_array = sizeof(T*) * lbatch_count;         \
    size_t const size_A_col_ptr_array = sizeof(T*) * lbatch_count;         \
    size_t const size_A_last_row_ptr_array = sizeof(T*) * 1 * batch_count; \
    size_t const size_A_last_col_ptr_array = sizeof(T*) * 1 * batch_count; \
    size_t const size_A_ptr_array = sizeof(T*) * batch_count;              \
                                                                           \
    T** A_row_ptr_array = (T**)pfree;                                      \
    pfree += size_A_row_ptr_array;                                         \
                                                                           \
    T** A_col_ptr_array = (T**)pfree;                                      \
    pfree += size_A_col_ptr_array;                                         \
                                                                           \
    T** A_last_row_ptr_array = (T**)pfree;                                 \
    pfree += size_A_last_row_ptr_array;                                    \
                                                                           \
    T** A_last_col_ptr_array = (T**)pfree;                                 \
    pfree += size_A_last_col_ptr_array;                                    \
                                                                           \
    T** A_ptr_array = (T**)pfree;                                          \
    pfree += size_A_ptr_array;                                             \
                                                                           \
    total_bytes += size_A_row_ptr_array;                                   \
    total_bytes += size_A_col_ptr_array;                                   \
    total_bytes += size_A_last_row_ptr_array;                              \
    total_bytes += size_A_last_col_ptr_array;                              \
    total_bytes += size_A_ptr_array;                                       \
                                                                           \
    size_t const size_A_diag_ptr_array = sizeof(T*) * lbatch_count;        \
    size_t const size_A_last_diag_ptr_array = sizeof(T*) * batch_count;    \
                                                                           \
    T** A_diag_ptr_array = (T**)pfree;                                     \
    pfree += size_A_diag_ptr_array;                                        \
                                                                           \
    T** A_last_diag_ptr_array = (T**)pfree;                                \
    pfree += size_A_last_diag_ptr_array;                                   \
                                                                           \
    total_bytes += size_A_diag_ptr_array;                                  \
    total_bytes += size_A_last_diag_ptr_array;

#define ALLOC_RESIDUAL_AJ()                                       \
    size_t const size_residual_Aj = sizeof(S) * lbatch_count;     \
    size_t const size_residual_Aj_last = sizeof(S) * batch_count; \
                                                                  \
    S* const residual_Aj = (S*)pfree;                             \
    pfree += size_residual_Aj;                                    \
                                                                  \
    S* const residual_Aj_last = (S*)pfree;                        \
    pfree += size_residual_Aj_last;                               \
                                                                  \
    total_bytes += size_residual_Aj;                              \
    total_bytes += size_residual_Aj_last;

#define ALLOC_INFO_AJ()                                       \
                                                              \
    size_t const size_info_Aj = sizeof(I) * lbatch_count;     \
    size_t const size_info_Aj_last = sizeof(I) * batch_count; \
                                                              \
    I* const info_Aj = (I*)pfree;                             \
    pfree += size_info_Aj;                                    \
                                                              \
    I* const info_Aj_last = (I*)pfree;                        \
    pfree += size_info_Aj_last;                               \
                                                              \
    total_bytes += size_info_Aj;                              \
    total_bytes += size_info_Aj_last;

#define ALLOC_W_AJ()                                                        \
                                                                            \
    size_t const size_W_Aj = sizeof(S) * (2 * nb) * lbatch_count;           \
    size_t const size_W_Aj_last = sizeof(S) * (nb + nb_last) * batch_count; \
                                                                            \
    S* const W_Aj = (S*)pfree;                                              \
    pfree += size_W_Aj;                                                     \
                                                                            \
    S* const W_Aj_last = (S*)pfree;                                         \
    pfree += size_W_Aj_last;                                                \
                                                                            \
    total_bytes += size_W_Aj;                                               \
    total_bytes += size_W_Aj_last;

#define ALLOC_EIG_MAP()                                                           \
    size_t const size_eig_map_bytes = (need_V) ? sizeof(I) * n * batch_count : 0; \
    I* const eig_map = (need_V) ? (I*)pfree : nullptr;                            \
    pfree += size_eig_map_bytes;                                                  \
    total_bytes += size_eig_map_bytes;

#define ALLOC_N_SWEEPS_AJ()                                       \
                                                                  \
    size_t const size_n_sweeps_Aj = sizeof(I) * lbatch_count;     \
    size_t const size_n_sweeps_Aj_last = sizeof(I) * batch_count; \
                                                                  \
    I* const n_sweeps_Aj = (I*)pfree;                             \
    pfree += size_n_sweeps_Aj;                                    \
                                                                  \
    I* const n_sweeps_Aj_last = (I*)pfree;                        \
    pfree += size_n_sweeps_Aj_last;                               \
                                                                  \
    total_bytes += size_n_sweeps_Aj;                              \
    total_bytes += size_n_sweeps_Aj_last;

#define ALLOC_WORK_ROCBLAS()                                                  \
                                                                              \
    size_t const size_work_rocblas = sizeof(T*) * (nblocks_half)*batch_count; \
                                                                              \
    T** const work_rocblas = (T**)pfree;                                      \
    pfree += size_work_rocblas;                                               \
                                                                              \
    total_bytes += size_work_rocblas;

#define ALLOC_MATE_ARRAY()                                            \
                                                                      \
    size_t const size_mate_array = sizeof(I) * nblocks * batch_count; \
                                                                      \
    I* const mate_array = (I*)pfree;                                  \
    pfree += size_mate_array;                                         \
                                                                      \
    total_bytes += size_mate_array;

#define ALLOC_GMAT()                                                        \
                                                                            \
    size_t const size_Gmat = sizeof(S) * (nblocks * nblocks) * batch_count; \
                                                                            \
    S* const Gmat = (S*)pfree;                                              \
    pfree += size_Gmat;                                                     \
                                                                            \
    total_bytes += size_Gmat;

#define ALLOC_SCHEDULE()                                                \
    I const nplayers_last = (nb + nb_last) + ((nb + nb_last) % 2);      \
    I const nplayers_small = (2 * nb);                                  \
    I const nplayers_large = even_nblocks;                              \
                                                                        \
    I const len_schedule_last = nplayers_last * (nplayers_last - 1);    \
    I const len_schedule_small = nplayers_small * (nplayers_small - 1); \
    I const len_schedule_large = nplayers_large * (nplayers_large - 1); \
                                                                        \
    size_t const size_schedule_last = sizeof(I) * len_schedule_last;    \
    size_t const size_schedule_small = sizeof(I) * len_schedule_small;  \
    size_t const size_schedule_large = sizeof(I) * len_schedule_large;  \
                                                                        \
    I* const d_schedule_last = (I*)pfree;                               \
    pfree += size_schedule_last;                                        \
                                                                        \
    I* const d_schedule_small = (I*)pfree;                              \
    pfree += size_schedule_small;                                       \
                                                                        \
    I* const d_schedule_large = (I*)pfree;                              \
    pfree += size_schedule_large;                                       \
                                                                        \
    total_bytes += size_schedule_last;                                  \
    total_bytes += size_schedule_small;                                 \
    total_bytes += size_schedule_large;

#define ALLOC_ALL()       \
    ALLOC_INIT();         \
    ALLOC_A();            \
    ALLOC_Atmp();         \
    ALLOC_Vtmp();         \
    ALLOC_RESIDUAL_AJ();  \
    ALLOC_INFO_AJ();      \
    ALLOC_W_AJ();         \
    ALLOC_N_SWEEPS_AJ();  \
    ALLOC_WORK_ROCBLAS(); \
    ALLOC_MATE_ARRAY();   \
    ALLOC_GMAT();         \
    ALLOC_AJ();           \
    ALLOC_VJ();           \
    ALLOC_EIG_MAP();      \
    ALLOC_SCHEDULE();

#define SWAP_Atmp_Vtmp()                                          \
    {                                                             \
        swap(Atmp, Vtmp);                                         \
        swap(Atmp_row_ptr_array, Vtmp_row_ptr_array);             \
        swap(Atmp_col_ptr_array, Vtmp_col_ptr_array);             \
        swap(Atmp_last_row_ptr_array, Vtmp_last_row_ptr_array);   \
        swap(Atmp_last_col_ptr_array, Vtmp_last_col_ptr_array);   \
        swap(Atmp_ptr_array, Vtmp_ptr_array);                     \
                                                                  \
        swap(Atmp_diag_ptr_array, Vtmp_diag_ptr_array);           \
        swap(Atmp_last_diag_ptr_array, Vtmp_last_diag_ptr_array); \
                                                                  \
        swap(lstride_Atmp, lstride_Vtmp);                         \
        swap(ldatmp, ldvtmp);                                     \
        swap(shift_Atmp, shift_Vtmp);                             \
    }

#define SWAP_AJ_VJ()                            \
    swap(Vj, Aj);                               \
    swap(Vj_last, Aj_last);                     \
    swap(Vj_ptr_array, Aj_ptr_array);           \
    swap(Vj_last_ptr_array, Aj_last_ptr_array); \
                                                \
    swap(ldvj, ldaj);                           \
    swap(ldvj_last, ldaj_last);                 \
    swap(lstride_Vj, lstride_Aj);               \
    swap(lstride_Vj_last, lstride_Aj_last);     \
    swap(shift_Vj, shift_Aj);                   \
    swap(shift_Vj_last, shift_Aj_last);

/************** CPU functions                              *******************/
/*****************************************************************************/
static size_t get_lds()
{
    size_t const default_lds_size = 64 * 1024;

    int lds_size = 0;
    int deviceId = 0;
    auto istat_device = hipGetDevice(&deviceId);
    if(istat_device != hipSuccess)
    {
        return (default_lds_size);
    };
    auto const attr = hipDeviceAttributeMaxSharedMemoryPerBlock;
    auto const istat_attr = hipDeviceGetAttribute(&lds_size, attr, deviceId);
    if(istat_attr != hipSuccess)
    {
        return (default_lds_size);
    };

    return (lds_size);
}

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

    std::vector<I> new2old(nplayers);
    std::vector<I> old2new(nplayers);

    for(I iround = 0; iround < num_rounds; iround++)
    {
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

// Max number of threads per thread-block used in rsyevj_small kernel
static constexpr auto NX_THREADS = 32;
static constexpr auto NY_THREADS = 32;
static constexpr auto RSYEVJ_BDIM = NX_THREADS * NY_THREADS;

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

template <typename T, typename I, typename S>
static __device__ void check_symmetry_body(I const n,
                                           T const* const A_,
                                           I const lda,
                                           I const i_start,
                                           I const i_inc,
                                           I const j_start,
                                           I const j_inc,
                                           int* n_unsymmetric,
                                           S tol = 10 * std::numeric_limits<S>::round_error())
{
    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * int64_t(ld)); };

    auto A = [=](auto i, auto j) -> T { return (A_[idx2D(i, j, lda)]); };

    auto nearly_equal = [=](auto x, auto y, auto tol) -> bool {
        auto const dnorm = std::max(std::abs(x), std::abs(y));
        return (std::abs(x - y) <= (tol * dnorm));
    };

    // ----------------------------------------
    // ** note ** atomicAnd works on type "int"
    // not on type "bool"
    // ----------------------------------------

    __syncthreads();

    for(auto j = j_start; j < n; j += j_inc)
    {
        for(auto i = i_start; i < n; i += i_inc)
        {
            auto const aij = A(i, j);
            auto const aji = A(j, i);
            if(!nearly_equal(aij, conj(aji), tol))
            {
                // -----------------------
                // matrix is not symmetric
                // -----------------------
                atomicAdd(n_unsymmetric, 1);
                break;
            }
        }
    }
    __syncthreads();
}

// -----------------------------------------
// check whether a matrix is symmetric
//
// assume n_unsymmetric[] is initialized to 0
//
// launch as dim3(1,1,nbz), dim3(nx,ny,1)
// -----------------------------------------
template <typename T, typename I, typename UA, typename Istride>
__global__ static void check_symmetry_kernel(I const n,
                                             UA A_,
                                             Istride const shiftA,
                                             I const lda,
                                             Istride const strideA,

                                             int n_unsymmetric[],
                                             I const batch_count)
{
    using S = decltype(std::real(T{}));

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const j_start = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;

    I const i_inc = hipBlockDim_x * hipGridDim_x;
    I const j_inc = hipBlockDim_y * hipGridDim_y;

    I const bid_inc = hipGridDim_z;
    I const bid_start = hipBlockIdx_z;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const Ap = load_ptr_batch(A_, bid, shiftA, strideA);

        check_symmetry_body<T, I, S>(n, Ap, lda, i_start, i_inc, j_start, j_inc,
                                     &(n_unsymmetric[bid]));
    }
}

template <typename T, typename I, typename UA, typename Istride>
static void check_symmetry(I const n,
                           UA A_,
                           Istride const shiftA,
                           I const lda,
                           Istride const strideA,
                           int n_unsymmetric[],
                           I const batch_count,
                           hipStream_t stream)
{
    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto const istat = hipMemsetAsync(&(n_unsymmetric[0]), 0, sizeof(I) * batch_count, stream);
    assert(istat == HIP_SUCCESS);

    auto const max_thread_blocks = 64 * 1000;
    auto const nx = NX_THREADS;
    auto const ny = NY_THREADS;
    auto const nbx = std::min(max_thread_blocks, ceil(n, nx));
    auto const nby = std::min(max_thread_blocks, ceil(n, ny));
    auto const nbz = std::min(max_thread_blocks, batch_count);

    check_symmetry_kernel<T, I, UA, Istride><<<dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream>>>(
        n, A_, shiftA, lda, strideA, n_unsymmetric, batch_count);
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

            if(is_diag)
            {
                auto const aii = A(i, i);
                A(i, i) = (aii + conj(aii)) / 2;
            }
            else
            {
                bool const do_assignment
                    = (use_upper && is_strictly_upper) || (use_lower && is_strictly_lower);

                if(do_assignment)
                {
                    A(j, i) = conj(A(i, j));
                }
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

        using S = decltype(std::real(T{}));
        S const ulp = 10;
        S const tol = ulp * std::numeric_limits<S>::round_error();

        bool is_symmetric = true;

        T aij_err = 0;
        T aji_err = 0;

        I ierr = 0;
        I jerr = 0;

        auto nearly_equal = [](auto x, auto y, auto tol) -> bool {
            auto const dnorm = std::max(std::abs(x), std::abs(y));
            return (std::abs(x - y) <= tol * dnorm);
        };

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
                    if(!nearly_equal(aij, conj(aji), tol))
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

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto A = [=](auto i, auto j) -> const T& { return (A_[idx2D(i, j, lda)]); };

    auto B = [=](auto i, auto j) -> T& { return (B_[idx2D(i, j, ldb)]); };

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
// laset set off-diagonal entries of m by n matrix to alpha
// and diagonal entries to beta
//
// if (uplo == 'U') set only the upper triangular part
// if (uplo == 'L') set only the lower triangular part
//
// if (uplo == 'A') set the whole m by n matrix
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
    bool const use_full = (uplo == 'A') || (uplo == 'a');

    bool const is_valid = use_upper || use_lower || use_full;
    assert(is_valid);

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

            if(is_diag)
            {
                A(i, i) = beta_diag;
            }
            else
            {
                bool const do_assignment = (use_full || (use_lower && is_strictly_lower)
                                            || (use_upper && is_strictly_upper));

                if(do_assignment)
                {
                    A(i, j) = alpha_offdiag;
                }
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
        I* map = (has_map) ? map_ + bid * stridemap : nullptr;

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
                                       I const* const row_map_,
                                       I const* const col_map_,
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
    bool const has_row_map = (row_map_ != nullptr);
    bool const has_col_map = (col_map_ != nullptr);

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

        I const* const row_map = (has_row_map) ? &(row_map_[bid * n]) : nullptr;
        I const* const col_map = (has_col_map) ? &(col_map_[bid * n]) : nullptr;

        auto const Ap = [=](auto i, auto j) -> const T& { return (A_p[idx2D(i, j, lda)]); };
        auto const Bp = [=](auto i, auto j) -> T& { return (B_p[idx2D(i, j, ldb)]); };

        for(auto j = j_start; j < n; j += j_inc)
        {
            for(auto i = i_start; i < m; i += i_inc)
            {
                auto const ia = (row_map != nullptr) ? row_map[i] : i;
                auto const ja = (col_map != nullptr) ? col_map[j] : j;

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
                                      I const* const row_map_,
                                      I const* const col_map_,
                                      Istride const stride_map,
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
    bool const has_row_map = (row_map_ != nullptr);
    bool const has_col_map = (col_map_ != nullptr);

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

#ifdef NDEBUG
#else
    // -----------------------------------------
    // check that map the last block also to the last block
    // -----------------------------------------
    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        bool const is_root = (i_start == 0) && (j_start == 0);

        if(has_row_map)
        {
            I const* const row_map = row_map_ + bid * stride_map;

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
            I const* const col_map = col_map_ + bid * stride_map;

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

#endif
    // ----------------------
    // size of the i-th block
    // ----------------------
    auto const nb_last = n - (nblocks - 1) * nb;
    auto bsize = [=](auto iblock) { return ((iblock == (nblocks - 1)) ? nb_last : nb); };

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const dA = load_ptr_batch<T>(AA, bid, shiftA, strideA);
        T* const dC = load_ptr_batch<T>(CC, bid, shiftC, strideC);

        I const* const row_map = (has_row_map) ? row_map_ + bid * stride_map : nullptr;
        I const* const col_map = (has_col_map) ? col_map_ + bid * stride_map : nullptr;

        for(auto jblock = jb_start; jblock < nblocks; jblock += jb_inc)
        {
            for(auto iblock = ib_start; iblock < nblocks; iblock += ib_inc)
            {
                auto const iblock_old = (row_map != nullptr) ? row_map[iblock] : iblock;
                auto const jblock_old = (col_map != nullptr) ? col_map[jblock] : jblock;

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
// dim3(nbx,nby,nbz), dim3(nx,ny,1)
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

/* 
 * -----------------------------------------------------------------
 * Greedy heuristic to solve the maximum weight matching on a graph
 * to pick the most profitable set of independent pairs
 * -----------------------------------------------------------------
 */
typedef int Icount;

template <typename T, typename I>
static void __device__ pgreedy_mwm_block(I const n, T const* const G_, I* const mate_)
{
    // Parallel greedy algorithm to find the maximum weight edge matching
    // of a complete graph
    // Assume the positive weights are in
    // n by n matrix symmetric G(0:(n-1), 0:(n-1))
    // Assume G(i,j) == G(j,i) and  (i,i) = 0
    //
    // The matching is returned in 2 by (n/2) array mate[]
    //

    auto is_even = [](auto n) -> bool { return ((n % 2) == 0); };

    // ------------------------
    // execute in a single block
    // ------------------------
    auto mate = [=](auto i, auto j) -> I& {
        auto const ij = i + j * 2;
        return (mate_[ij]);
    };

    if(n <= 0)
    {
        return;
    };

    assert(is_even(n));

    auto G = [=](auto i, auto j) -> const T {
        auto const ij = i + j * I(n);
        return ((i == j) ? (0) : G_[ij]);
    };

    // ------------------------
    // for each vertex i,  iwmax(i) holds
    // the available vertex with max weight
    // note the use of Icount
    // ------------------------

    extern __shared__ double lmem[];
    __shared__ int nmate;

    std::byte* pfree = (std::byte*)lmem;
    Icount* iwmax = (Icount*)pfree;
    pfree += sizeof(Icount) * n;
    bool* is_matched = (bool*)pfree;
    pfree += sizeof(bool) * n;

    auto const tid = hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x
        + hipThreadIdx_z * (hipBlockDim_x * hipBlockDim_y);
    auto const nthreads = (hipBlockDim_x * hipBlockDim_y) * hipBlockDim_z;

    auto const i_start = tid;
    auto const i_inc = nthreads;

    auto max = [](auto x, auto y) { return ((x > y) ? x : y); };

    auto min = [](auto x, auto y) { return ((x < y) ? x : y); };

    auto swap = [=](auto& x, auto& y) {
        auto const t = x;
        x = y;
        y = t;
    };

    if(i_start == 0)
    {
        nmate = 0;
    }

    Icount const iwmax_invalid = n;

    __syncthreads();

    for(I ij = i_start; ij < n; ij += i_inc)
    {
        is_matched[ij] = false;
        iwmax[ij] = iwmax_invalid;
    }

    __syncthreads();

    I ipass = 0;
    for(ipass = 0; ipass < 2 * (n * n); ipass++)
    {
        bool const is_done = (nmate >= (n / 2));
        if(is_done)
        {
            break;
        };
        __syncthreads();

        // -------------------------------------
        // for each available vertex j, compute
        // the matching vertex with max weight
        // -------------------------------------

        for(I jvertex = i_start; jvertex < n; jvertex += i_inc)
        {
            if(is_matched[jvertex])
            {
                continue;
            }

            // ----------------------------------------
            // find the neighbor vertex with max weight
            // ----------------------------------------

            iwmax[jvertex] = iwmax_invalid;
            T wmax = std::numeric_limits<T>::lowest();
            for(I ivertex = 0; ivertex < n; ivertex++)
            {
                if(is_matched[ivertex])
                {
                    continue;
                }
                if(ivertex == jvertex)
                {
                    continue;
                };

                // ---------------------------------
                // break ties by vertex label number
                // ---------------------------------
                T const Gij = (G(jvertex, ivertex) + G(ivertex, jvertex)) / 2;
                bool const is_greater = (Gij > wmax) || ((Gij == wmax) && (ivertex < iwmax[jvertex]));
                if(is_greater)
                {
                    wmax = Gij;
                    iwmax[jvertex] = ivertex;
                }
            }
        }
        __syncthreads();

        // -------------------------------------------------------------
        // for each local max edge, check whether it is also locally max
        // -------------------------------------------------------------

        for(I i = i_start; i < n; i += i_inc)
        {
            if(is_matched[i])
            {
                continue;
            };
            auto const j = iwmax[i];

            bool const isok_j = (0 <= j) && (j < n);
            bool const is_accept = isok_j && (!is_matched[j]) && (iwmax[j] == i) && (i > j);
            if(is_accept)
            {
                auto const min_ij = min(i, j);
                auto const max_ij = max(i, j);

                is_matched[i] = true;
                is_matched[j] = true;

                auto const ip = atomicAdd(&nmate, 1);

                mate(0, ip) = min_ij;
                mate(1, ip) = max_ij;
            }
        }

        __syncthreads();

    } // end for (ipass)

    __syncthreads();

    {
        bool const isok = (nmate == (n / 2));
        if(!isok)
        {
            if(i_start == 0)
            {
                printf("pgreedy: nmate=%d, n=%d\n", nmate, n);
                for(I iv = 0; iv < n; iv++)
                {
                    auto const jv = iwmax[iv];
                    printf("is_matched[%d]=%d,iwmax[%d]=%d, G(%d,%d)=%le, G(%d,%d)=%le\n", (int)iv,
                           (int)is_matched[iv], (int)iv, (int)iwmax[iv], (int)iv, (int)jv,
                           G(iv, jv), (int)jv, (int)iv, G(jv, iv));
                }
            }
        }
        __syncthreads();
        assert(nmate == (n / 2));
    }

    // ------------------------
    // place last entry  at last position
    // ------------------------

    for(auto i = i_start; i < (nmate - 1); i += i_inc)
    {
        auto const ivertex = mate(0, i);
        auto const jvertex = mate(1, i);
        bool const is_need_swap = (max(ivertex, jvertex) == (n - 1));
        if(is_need_swap)
        {
            swap(mate(0, i), mate(0, (nmate - 1)));
            swap(mate(1, i), mate(1, (nmate - 1)));
        }
    }
    __syncthreads();

#ifdef NDEBUG
#else
    {
        // check the last entry
        assert(mate(1, nmate - 1) == (n - 1));

        // -------------------------------------------
        // check mat(0:1,0:(nmate-1)) is a permutation
        //
        // reuse storage for iwmax(:)
        // -------------------------------------------
        __syncthreads();

        for(I i = i_start; i < n; i += i_inc)
        {
            iwmax[i] = 0;
        }

        __syncthreads();

        for(I i = i_start; i < nmate; i += i_inc)
        {
            auto const iv = mate(0, i);
            auto const jv = mate(1, i);
            atomicAdd(&(iwmax[iv]), 1);
            atomicAdd(&(iwmax[jv]), 1);
        }
        __syncthreads();

        for(I i = i_start; i < n; i += i_inc)
        {
            assert(iwmax[i] == 1);
        }

        __syncthreads();
    }
#endif
}

template <typename T, typename I>
static void __global__
    pgreedy_mwm_kernel(I const n, T const* const G_array, I* const mate_array, I const batch_count)
{
    auto bid_start = hipBlockIdx_x + hipBlockIdx_y * (hipGridDim_x)
        + hipBlockIdx_z * (hipGridDim_x * hipGridDim_y);

    auto bid_inc = (hipGridDim_x * hipGridDim_y) * hipGridDim_z;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const G_ = &(G_array[bid * (n * int64_t(n))]);
        I* const mate = &(mate_array[bid * n]);

        pgreedy_mwm_block(n, G_, mate);
    }
}

template <typename T, typename I>
static void pgreedy_mwm(I const n,
                        T const* const G_array,
                        I* const mate_array,
                        I const batch_count,
                        hipStream_t stream)
{
    if(n <= 0)
    {
        return;
    };

    auto const MAX_THREADS = 1024;
    auto const MAX_BLOCKS = 64 * 1000;

    auto const nwarps = (n - 1) / warpSize + 1;
    auto const nthreads = std::min(nwarps * warpSize, MAX_THREADS);

    auto const nblocks = std::min(MAX_BLOCKS, batch_count);

    size_t const lmem_size = n * (sizeof(Icount) + sizeof(bool));
    size_t const max_lds = get_lds();
    assert(lmem_size + sizeof(int) <= max_lds);

    pgreedy_mwm_kernel<T, I><<<dim3(1, 1, nblocks), dim3(1, 1, 1), lmem_size, stream>>>(
        n, G_array, mate_array, batch_count);
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
    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    auto A = [=](auto i, auto j) -> const T& { return (A_[idx2D(i, j, lda)]); };

    bool const is_root = (i_start == 0) && (j_start == 0);

    S const zero = 0.0;
    constexpr bool use_serial = true;

    if(use_serial)
    {
        // ------------------------------
        // simple serial code should work
        // ------------------------------

        if(is_root)
        {
            double dsum = 0.0;
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
    for(I i = 0; i < n; i++)
    {
        if(h_ptr_array[i] == nullptr)
        {
            nerr++;
        }
    }
    if(nerr != 0)
    {
        printf("check_ptr_array:%s, n = %d\n", msg.c_str(), n);
        fflush(stdout);

        for(I i = 0; i < n; i++)
        {
            if(h_ptr_array[i] == nullptr)
            {
                printf("%s[%d] is nullptr\n", msg.c_str(), i);
            }
        }

        fflush(stdout);
    }
}
#endif

/** kernel to setup pointer arrays in preparation
 * for calls to batched GEMM and for copying data
 *
 * launch as dim3(1,1,batch_count), dim3(nx,1,1)
**/
template <typename T, typename I, typename Istride, typename AA>
__global__ static void setup_ptr_arrays_kernel(

    bool const need_V,
    I const n,
    I const nb,

    AA A,
    Istride const shiftA,
    I const lda,
    Istride const strideA,

    T* const Atmp,
    Istride const shift_Atmp,
    I const ldatmp,
    Istride const lstride_Atmp,

    T* const Vtmp,
    Istride const shift_Vtmp,
    I const ldvtmp,
    Istride const lstride_Vtmp,

    T* const Aj,
    T* const Vj,
    T* const Aj_last,
    T* const Vj_last,

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

    T* Atmp_diag_ptr_array[],
    T* Atmp_last_diag_ptr_array[],

    T* Vtmp_diag_ptr_array[],
    T* Vtmp_last_diag_ptr_array[],

    I* const completed,
    I const batch_count)
{
    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_start = hipThreadIdx_x;
    I const i_inc = hipBlockDim_x;

    bool const is_root = (i_start == 0);

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto const nblocks = ceil(n, nb);
    auto const nb_last = n - (nblocks - 1) * nb;

    auto const nblocks_even = nblocks + (nblocks % 2);
    auto const nblocks_half = nblocks_even / 2;

    auto const idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    __shared__ I ibatch;

    if(is_root)
    {
        ibatch = 0;
    };
    __syncthreads();

    for(auto bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        bool const is_completed = completed[bid + 1];
        if(is_completed)
        {
            continue;
        };

        T* const A_p = load_ptr_batch(A, bid, shiftA, strideA);
        T* const Atmp_p = load_ptr_batch(Atmp, bid, shift_Atmp, lstride_Atmp);
        T* const Vtmp_p = (need_V) ? load_ptr_batch(Vtmp, bid, shift_Vtmp, lstride_Vtmp) : nullptr;

        if(is_root)
        {
            ibatch = atomicAdd(&(completed[0]), 1);
        }
        __syncthreads();

        A_ptr_array[ibatch] = A_p;
        Atmp_ptr_array[ibatch] = Atmp_p;
        Vtmp_ptr_array[ibatch] = (need_V) ? Vtmp_p : nullptr;

        auto const ilast_row = (nblocks_half - 1) * (2 * nb);
        auto const jlast_col = (nblocks_half - 1) * (2 * nb);

        A_last_row_ptr_array[ibatch] = A_p + idx2D(ilast_row, 0, lda);
        A_last_col_ptr_array[ibatch] = A_p + idx2D(0, jlast_col, lda);
        A_last_diag_ptr_array[ibatch] = A_p + idx2D(ilast_row, jlast_col, lda);

        Atmp_last_row_ptr_array[ibatch] = Atmp_p + idx2D(ilast_row, 0, ldatmp);
        Atmp_last_col_ptr_array[ibatch] = Atmp_p + idx2D(0, jlast_col, ldatmp);
        Atmp_last_diag_ptr_array[ibatch] = Atmp_p + idx2D(ilast_row, jlast_col, ldatmp);

        Vtmp_last_row_ptr_array[ibatch] = (need_V) ? Vtmp_p + idx2D(ilast_row, 0, ldvtmp) : nullptr;
        Vtmp_last_col_ptr_array[ibatch] = (need_V) ? Vtmp_p + idx2D(0, jlast_col, ldvtmp) : nullptr;
        Vtmp_last_diag_ptr_array[ibatch]
            = (need_V) ? Vtmp_p + idx2D(ilast_row, jlast_col, ldvtmp) : nullptr;

        {
            I const ldvj_last = (nb + nb_last);
            I const ldaj_last = (nb + nb_last);
            Vj_last_ptr_array[ibatch] = Vj_last + ibatch * ldvj_last * (nb + nb_last);
            Aj_last_ptr_array[ibatch] = Aj_last + ibatch * ldaj_last * (nb + nb_last);

            for(auto i = i_start; i < (nblocks_half - 1); i += i_inc)
            {
                Istride const stride_Aj
                    = static_cast<Istride>(nblocks_half - 1) * (2 * nb) * (2 * nb);
                Istride const stride_Vj
                    = static_cast<Istride>(nblocks_half - 1) * (2 * nb) * (2 * nb);

                auto const ip = i + ibatch * (nblocks_half - 1);
                Vj_ptr_array[ip] = Vj + ibatch * stride_Vj + i * (2 * nb) * (2 * nb);
                Aj_ptr_array[ip] = Aj + ibatch * stride_Aj + i * (2 * nb) * (2 * nb);
            }
        }

        for(auto i = i_start; i < (nblocks_half - 1); i += i_inc)
        {
            auto const ip = i + ibatch * (nblocks_half - 1);

            // ----------------
            // set row pointers
            // ----------------

            I const irow = i * (2 * nb);
            I const jcol = i * (2 * nb);

            A_row_ptr_array[ip] = A_p + idx2D(irow, 0, lda);
            Atmp_row_ptr_array[ip] = Atmp_p + idx2D(irow, 0, ldatmp);
            Vtmp_row_ptr_array[ip] = (need_V) ? Vtmp_p + idx2D(irow, 0, ldvtmp) : nullptr;

            // ----------------
            // set col pointers
            // ----------------

            A_col_ptr_array[ip] = A_p + idx2D(0, jcol, lda);
            Atmp_col_ptr_array[ip] = Atmp_p + idx2D(0, jcol, ldatmp);
            Vtmp_col_ptr_array[ip] = (need_V) ? Vtmp_p + idx2D(0, jcol, ldvtmp) : nullptr;

            // -----------------
            // set diag pointers
            // -----------------

            A_diag_ptr_array[ip] = A_p + idx2D(irow, jcol, lda);
            Atmp_diag_ptr_array[ip] = Atmp_p + idx2D(irow, jcol, ldatmp);
            Vtmp_diag_ptr_array[ip] = (need_V) ? Vtmp_p + idx2D(irow, jcol, ldvtmp) : nullptr;

        } // for i

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
    bool const need_V = (evect != rocblas_evect_none);

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
        int n_unsymmetric = 0;
        check_symmetry_body<T, I, S>(n, A_, ld1, i_start, i_inc, j_start, j_inc, &n_unsymmetric);

        bool const is_symmetric = (n_unsymmetric == 0);
        assert(is_symmetric);
    }
#endif

    S norm_offdiag = 0;
#ifdef NDEBUG
#else
    if(idebug >= 1)
    {
        S* const Swork = (S*)dwork;
        bool const need_diagonal = false;
        cal_norm_body(n, n, A_, lda, Swork, i_start, i_inc, j_start, j_inc, need_diagonal);
        norm_offdiag = Swork[0];
        if((i_start == 0) && (j_start == 0))
        {
            printf("initial norm_offdiag=%le\n", norm_offdiag);
        }
    }

#endif
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
            __syncthreads();

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
    assert(n <= (nblocks * nb));

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
        T const* const Ap_ = load_ptr_batch(A, bid, shiftA, strideA);

        auto Ap = [=](auto i, auto j) -> const T& { return (Ap_[idx2D(i, j, lda)]); };

        for(I jb = jb_start; jb < nblocks; jb += jb_inc)
        {
            for(I ib = ib_start; ib < nblocks; ib += ib_inc)
            {
                bool const is_diag_block = (ib == jb);

                bool const need_diagonal
                    = (is_diag_block && include_diagonal_values) || (!is_diag_block);

                // -------------------------------
                // Block (ib,jb) has size ni by nj
                // -------------------------------
                auto const ni = bsize(ib);
                auto const nj = bsize(jb);

                auto const ii = ib * nb;
                auto const jj = jb * nb;

                // -----------------------------------------------
                // compute norm of Block (ib,jb) submatrix block
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
__global__ static void sum_Gmat_kernel(I const n,
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
            n_sweeps[bid] = (n_sweeps[bid] == 0) ? h_sweeps : std::min(n_sweeps[bid], h_sweeps);

            atomicAdd(&(completed[0]), 1);
        };
        // debug
        printf("bid=%d,anorm=%le,gnorm=%le,abstol=%le,is_completed=%d,completed[0]=%d\n", bid,
               (double)anorm, (double)gnorm, (double)abstol, (int)is_completed, (int)completed[0]);
    }
    __syncthreads();
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

    bool const need_sort = (esort != rocblas_esort_none);
    bool const need_V = (evect != rocblas_evect_none);

    // check whether to use A_ and V_ in LDS

    // ----------------------------------------------------------
    // array cosine also used in comuputing matrix Frobenius norm
    // ----------------------------------------------------------
    size_t const size_cosine = sizeof(S) * n;
    size_t const size_sine = sizeof(T) * ntables;

    size_t const size_A = sizeof(T) * n * n;
    size_t const size_V = (need_V) ? sizeof(T) * n * n : 0;

    size_t const total_size = size_cosine + size_sine + size_A + size_V;

    size_t const max_lds = 64 * 1000;
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

            if(do_overwrite_A)
            {
                lacpy_body(c_uplo, mm, nn, V_, ld1, dA, ld2, i_start, i_inc, j_start, j_inc);
            }
            else
            {
                if(V_ != dV)
                {
                    lacpy_body(c_uplo, mm, nn, V_, ld1, dV, ldv, i_start, i_inc, j_start, j_inc);
                }
            }
        }

        __syncthreads();
    }
}

/****** Template function, workspace size and argument validation **********/
/***************************************************************************/

/** Helper to calculate workspace sizes **/
template <bool BATCHED, typename T, typename I, typename S, typename Istride = rocblas_stride>
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
        std::byte* pfree = nullptr;
        ALLOC_ALL();
    }

    *size_dwork_byte = total_bytes;
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
    ROCSOLVER_ENTER("rsyevj_rheevj_template", "esort:", esort, "evect:", evect, "uplo:", uplo,
                    "n:", n, "shiftA:", shiftA, "lda:", lda, "abstol:", abstol,
                    "max_sweeps:", max_sweeps, "bc:", batch_count);

    // quick return
    bool const has_work = (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return rocblas_status_success;
    }

    bool const need_sort = (esort != rocblas_esort_none);
    bool const need_V = (evect != rocblas_evect_none);

    Istride const shift_zero = 0;

    auto Atmp = Acpy;
    Istride shift_Atmp = shift_zero;
    auto ldatmp = n;
    Istride lstride_Atmp = ldatmp * n;

    auto Vtmp = J;
    Istride shift_Vtmp = shift_zero;
    I ldvtmp = n;
    Istride lstride_Vtmp = ldvtmp * n;

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
    size_t total_bytes = 0;

    // absolute tolerance for evaluating when the algorithm has converged
    S const eps = get_epsilon<S>();
    S const atol = (abstol <= 0 ? eps : abstol);

    // local variables
    I const even_n = n + (n % 2);
    I const n_even = even_n;
    I const half_n = even_n / 2;

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

        size_t const size_schedule_small = sizeof(I) * len_schedule_small;

        I* const d_schedule_small = (I*)pfree;
        pfree += size_schedule_small;

        total_bytes += size_schedule_small;

        assert(total_bytes <= size_dwork_byte);

        setup_schedule(nplayers, h_schedule_small, d_schedule_small, stream);

        {
            size_t const lmemsize = get_lds();
            Istride const strideAcpy = Istride(n) * n;
            ROCSOLVER_LAUNCH_KERNEL((rsyevj_small_kernel<T, I, S, U, Istride>),
                                    dim3(1, 1, batch_count), dim3(ddx, ddy, 1), lmemsize, stream,
                                    esort, evect, uplo, n, A, shiftA, lda, strideA, atol, eps,
                                    residual, max_sweeps, n_sweeps, W, strideW, info, Acpy,
                                    strideAcpy, batch_count, d_schedule_small);
        }
    }
    else
    {
        S* const Amat_norm = norms;

        ALLOC_ALL();
        assert(total_bytes <= size_dwork_byte);

        // ---------------------
        // launch configurations
        // ---------------------
        auto const max_lds = get_lds();
        auto const max_thread_blocks = 64 * 1000;
        auto const nx = NX_THREADS;
        auto const ny = RSYEVJ_BDIM / nx;

        auto const nbx = std::min(max_thread_blocks, ceil(n, nx));
        auto const nby = std::min(max_thread_blocks, ceil(n, ny));
        auto const nbz = std::min(max_thread_blocks, batch_count);

        auto update_norm = [=](bool const include_diagonal, S* residual) {
            // clang-format off
                ROCSOLVER_LAUNCH_KERNEL((cal_Gmat_kernel<T, I, S, Istride, U>), 
				dim3(1, 1, nbz), dim3(nx, ny, 1), 0, stream,
				n, nb, 
				A, shiftA, lda, strideA, 
				Gmat, 
				include_diagonal, completed, batch_count);
            // clang-format on

            size_t const shmem_size = sizeof(S);
            ROCSOLVER_LAUNCH_KERNEL((sum_Gmat_kernel<S, I>), dim3(1, 1, nbz), dim3(nx, ny, 1),
                                    shmem_size, stream, n, nb, Gmat, residual, completed,
                                    batch_count);
        };
#ifdef NDEBUG
#else

        auto print_residual = [=]() {
            std::vector<S> h_residual(batch_count);
            auto const istat1 = (hipMemcpyAsync(&(h_residual[0]), residual, sizeof(S) * batch_count,
                                                hipMemcpyDeviceToHost, stream));
            assert(istat1 == hipSuccess);

            auto const istat2 = (hipStreamSynchronize(stream));
            assert(istat2 == hipSuccess);

            for(I bid = 0; bid < batch_count; bid++)
            {
                printf("h_residual[%d] = %le\n", (int)bid, (double)h_residual[bid]);
            }
        };

        auto print_Gmat = [=]() {
            std::vector<S> h_Gmat(nblocks * nblocks * batch_count);
            size_t const nbytes = sizeof(S) * nblocks * nblocks * batch_count;
            auto const istat = hipMemcpy(&(h_Gmat[0]), Gmat, nbytes, hipMemcpyDeviceToHost);
            assert(istat == hipSuccess);

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
        };

        auto print_mate = [=]() {
            size_t const len_mate_array = size_mate_array / sizeof(I);

            std::vector<I> h_mate_array(len_mate_array);
            auto const istat1 = hipMemcpyAsync(&(h_mate_array[0]), mate_array, size_mate_array,
                                               hipMemcpyDeviceToHost, stream);
            assert(istat1 == hipSuccess);
            auto const istat2 = hipStreamSynchronize(stream);
            assert(istat2 == hipSuccess);

            auto h_mate = [=](auto i, auto j, auto bid) {
                return (h_mate_array[i + j * 2 + bid * nblocks]);
            };

            for(I bid = 0; bid < batch_count; bid++)
            {
                for(I imate = 0; imate < (nblocks_half); imate++)
                {
                    printf("mate(%d,%d) = (%d,%d)\n", (int)imate, (int)bid,
                           (int)h_mate(0, imate, bid), (int)h_mate(1, imate, bid));
                }
            }
        };
#endif

#ifdef NDEBUG
#else

        auto print_eig = [=](I const n, S* W, Istride const strideW, I const batch_count,
                             bool const is_summary = false) {
            // clang-format off
            ROCSOLVER_LAUNCH_KERNEL((copy_diagonal_kernel<T, I, U, Istride>), 
			    dim3(1, 1, batch_count), dim3(32, 32, 1), 0, stream, 
			    n, A, shiftA, lda, strideA, 
			    W, strideW, 
			    batch_count);
            // clang-format on

            std::vector<S> h_W(n * batch_count);

            for(I bid = 0; bid < batch_count; bid++)
            {
                auto const istat = (hipMemcpyAsync(&(h_W[bid * n]), &(W[bid * strideW]),
                                                   sizeof(S) * n, hipMemcpyDeviceToHost, stream));
                assert(istat == hipSuccess);
            }
            auto const istat = (hipStreamSynchronize(stream));
            assert(istat == hipSuccess);

            for(I bid = 0; bid < batch_count; bid++)
            {
                if(is_summary)
                {
                    double w_max = 0;
                    double w_min = 0;
                    double w_norm = 0;
                    for(I i = 0; i < n; i++)
                    {
                        double const w_i = h_W[i + bid * n];
                        w_max = std::max(w_max, w_i);
                        w_min = std::min(w_min, w_i);
                        w_norm += std::norm(w_i);
                    }
                    w_norm = std::sqrt(w_norm);

                    printf("w_max = %le, w_min = %le, w_norm = %le\n", w_max, w_min, w_norm);
                }
                else
                {
                    for(I i = 0; i < n; i++)
                    {
                        double const w_i = h_W[i + bid * n];
                        printf("W(%d,%d) = %le\n", (int)i, (int)bid, (double)w_i);
                    }
                }
            }
        };

        auto print_eig_summary = [=](auto const n, S* W, Istride const strideW, I const batch_count) {
            bool const is_summary = true;
            print_eig(n, W, strideW, batch_count, is_summary);
        };
#endif
        bool const use_adjust_schedule_large = false;
        {
            std::vector<I> h_schedule_last(len_schedule_last);
            std::vector<I> h_schedule_small(len_schedule_small);
            std::vector<I> h_schedule_large(len_schedule_large);

            bool const use_adjust_schedule_small = false;
            setup_schedule(nplayers_small, h_schedule_small, d_schedule_small, stream,
                           use_adjust_schedule_small);

            bool const use_adjust_schedule_last = false;
            setup_schedule(nplayers_last, h_schedule_last, d_schedule_last, stream,
                           use_adjust_schedule_last);

            setup_schedule(nplayers_large, h_schedule_large, d_schedule_large, stream,
                           use_adjust_schedule_large);
        }

        char const c_uplo = (uplo == rocblas_fill_upper) ? 'U'
            : (uplo == rocblas_fill_lower)               ? 'L'
                                                         : 'A';
        {
            // ---------------------------
            // make matrix to be symmetric
            // ---------------------------

            // clang-format off
            ROCSOLVER_LAUNCH_KERNEL((symmetrize_matrix_kernel<T, I, U, Istride>),
                                    dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream, 
				    c_uplo, n, 
				    A, shiftA, lda, strideA, 
				    batch_count);
            // clang-format on
        }

        // ----------------------------------
        // precompute norms of orginal matrix
        // ----------------------------------
        {
            bool const include_diagonal = true;
            update_norm(include_diagonal, Amat_norm);
        }
#ifdef NDEBUG
#else
        if(idebug >= 1)
        {
            std::vector<S> h_Amat_norm(batch_count);

            HIP_CHECK(hipMemcpyAsync(&(h_Amat_norm[0]), Amat_norm, sizeof(S) * batch_count,
                                     hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            for(I bid = 0; bid < batch_count; bid++)
            {
                printf("Amat_norm[%d] = %le\n", bid, (double)h_Amat_norm[bid]);
            }
        }
#endif

        I n_completed = 0;
        I h_sweeps = 0;
        bool is_converged = false;

        for(h_sweeps = 0; (h_sweeps < max_sweeps) && (!is_converged); h_sweeps++)
        {
            for(I iround = 0; iround < num_rounds; iround++)
            {
                /**
 * Main algorithm in each iteration:
 *
 * Note Atmp is a temporary copy of A
 *      Vtmp holds the eigenvectors of A if eigenvectors are requested
 *
 * (1) find set of independent pairs, this may be computed from
 *     greedy algorithm for solving the Maximum Weighted Matching (MWM)
 *     based on norms computed in Gmat(nblocks,nblocks,batch_count_remain)
 *     or
 *     from lookup table  d_schedule_large( nblocks, num_rounds )
 *
 * (2) symmetric reordering in  copying from A to Atmp
 *     so that the independent pairs are (0,1),(2,3),(4,5)...
 *
 *     reorder block columns of Vtmp
 *     Atmp <-  Vtmp * P
 *     swap( Atmp, Vtmp )
 *
 *     Atmp <-  P' * A * P
 *
 *
 * (3) copy the pairs of diagonal blocks into 
 *     Aj( (2*nb),(2*nb),(nblocks-1),batch_count_remain)
 *
 * (4) perform Jacobi method in Aj() and store the
 *     eigenvectors in Vj( (2*nb),(2*nb),(nblocks-1),batch_count_remain )
 *
 * (5) Apply transformations from Vj to block rows of Atmp
 *     A <- Vj' * Atmp
 *
 * (6) Apply transformations from Vj to block cols of A
 *     Atmp <- A * Vj
 *
 * (6b) optional: copy diagonal blocks from Aj to Atmp
 *
 *
 * (7) Copy Atmp -> A for next iteration
 *
 * (8) Apply transformation from Vj to Vtmp
 *     Atmp <-  Vtmp * Vj
 *
 * (9) swap(Atmp,Vtmp)
 *
**/

#ifdef NDEBUG
#else
                if(idebug >= 1)
                {
                    I* const n_unsymmetric = info_Aj;

                    check_symmetry<T, I, U, Istride>(n, A, shiftA, lda, strideA, n_unsymmetric,
                                                     batch_count, stream);

                    std::vector<I> h_n_unsymmetric(batch_count);
                    auto const istat = hipMemcpy(&(h_n_unsymmetric[0]), n_unsymmetric,
                                                 sizeof(I) * batch_count, hipMemcpyDeviceToHost);

                    for(auto bid = 0; bid < batch_count; bid++)
                    {
                        bool const is_symmetric = (h_n_unsymmetric[bid] == 0);
                        if(!is_symmetric)
                        {
                            printf("(%d,%d), matrix A[%d] is non-symmetric\n", h_sweeps, iround, bid);
                        }
                    }
                }

#endif

                // -----------------------------------------------------
                // compute norms of blocks into array
                //
                // Gmat(0:(nblocks-1), 0:(nblocks-1), 0:(batch_count-1))
                // -----------------------------------------------------

                {
                    bool const include_diagonal = false;
                    update_norm(include_diagonal, residual);
                }

#ifdef NDEGBUG
#else
                if(idebug >= 1)
                {
                    TRACE(1);
                    print_Gmat();
                    print_residual();
                }
#endif

                {
                    // --------------------------------------------
                    // zero out just complete[0] to count number of
                    // completed batch entries
                    // --------------------------------------------
                    int const ivalue = 0;
                    size_t const nbytes = sizeof(I);
                    HIP_CHECK(hipMemsetAsync(&(completed[0]), ivalue, nbytes, stream));

                    auto const nnx = 64;
                    auto const nnb = std::min(max_thread_blocks, ceil(batch_count, nnx));

                    // clang-format off
                    ROCSOLVER_LAUNCH_KERNEL((set_completed_kernel<S, I, Istride>), 
				    dim3(1, 1, 1), dim3(1, 1, 1), 0, stream, 
				    n, nb, Amat_norm, atol,
				    h_sweeps, n_sweeps, residual, info, completed,
				    batch_count);
                    // clang-format on
                }

                {
                    // --------------------------------------
                    // check convergence of all batch entries
                    // --------------------------------------
                    HIP_CHECK(hipMemcpyAsync(&(n_completed), &(completed[0]), sizeof(I),
                                             hipMemcpyDeviceToHost, stream));
                    HIP_CHECK(hipStreamSynchronize(stream));

                    is_converged = (n_completed >= batch_count);

#ifdef NDEBUG
#else
                    if(idebug >= 1)
                    {
                        printf("n_completed=%d,batch_count=%d\n", (int)n_completed, (int)batch_count);
                    }
#endif

                    if(is_converged)
                    {
                        break;
                    };
                }

                auto const batch_count_remain = batch_count - n_completed;
                assert((1 <= batch_count_remain) && (batch_count_remain <= batch_count));
                TRACE(2);
                {
                    if(idebug >= 1)
                    {
                        printf("h_sweeps=%d, iround=%d, n=%d, nb=%d, nblocks=%d, "
                               "batch_count_remain=%d\n",
                               h_sweeps, iround, n, nb, nblocks, batch_count_remain);
                        fflush(stdout);
                    }
                }

                {
                    // ------------------------------------------
                    // build pointer arrays for data movement and
                    // for rocblas batch GEMM operations
                    // ------------------------------------------

                    // ---------------------------------
                    // reset value to be used as counter
                    // ---------------------------------
                    HIP_CHECK(hipMemsetAsync((void*)&(completed[0]), 0, sizeof(I), stream));

                    TRACE(2);
                    auto const nx = NX_THREADS;

                    // clang-format off
                    ROCSOLVER_LAUNCH_KERNEL(
                        (setup_ptr_arrays_kernel<T, I, Istride, U>), 

			dim3(1, 1, 1), dim3(1, 1, 1), 0, stream,

                        need_V, n, nb,

                        A, shiftA, lda, strideA,

                        Atmp, shift_Atmp, ldatmp, lstride_Atmp,

                        Vtmp, shift_Vtmp, ldvtmp, lstride_Vtmp,

                        Aj, Vj, Aj_last, Vj_last,

                        Vj_ptr_array, Aj_ptr_array, Vj_last_ptr_array, Aj_last_ptr_array,

                        A_row_ptr_array, A_col_ptr_array, A_last_row_ptr_array,
                        A_last_col_ptr_array, A_ptr_array,

                        Atmp_row_ptr_array, Atmp_col_ptr_array, Atmp_last_row_ptr_array,
                        Atmp_last_col_ptr_array, Atmp_ptr_array,

                        Vtmp_row_ptr_array, Vtmp_col_ptr_array, Vtmp_last_row_ptr_array,
                        Vtmp_last_col_ptr_array, Vtmp_ptr_array,

                        A_diag_ptr_array, A_last_diag_ptr_array, Atmp_diag_ptr_array,
                        Atmp_last_diag_ptr_array, Vtmp_diag_ptr_array, Vtmp_last_diag_ptr_array,

                        completed, batch_count);
                    // clang-format on

#ifdef NDEBUG
#else

                    if((idebug >= 1) && (batch_count_remain >= 1))
                    {
                        auto const lbatch_count = (nblocks_half - 1) * batch_count_remain;

                        check_ptr_array("Aj_last_ptr_array", batch_count_remain, Aj_last_ptr_array);
                        check_ptr_array("Vj_last_ptr_array", batch_count_remain, Vj_last_ptr_array);

                        check_ptr_array("A_last_row_ptr_array", batch_count_remain,
                                        A_last_row_ptr_array);
                        check_ptr_array("A_last_col_ptr_array", batch_count_remain,
                                        A_last_col_ptr_array);
                        check_ptr_array("A_last_diag_ptr_array", batch_count_remain,
                                        A_last_diag_ptr_array);

                        check_ptr_array("Atmp_last_row_ptr_array", batch_count_remain,
                                        Atmp_last_row_ptr_array);
                        check_ptr_array("Atmp_last_col_ptr_array", batch_count_remain,
                                        Atmp_last_col_ptr_array);
                        check_ptr_array("Atmp_last_diag_ptr_array", batch_count_remain,
                                        Atmp_last_diag_ptr_array);

                        if(need_V)
                        {
                            check_ptr_array("Vtmp_last_row_ptr_array", batch_count_remain,
                                            Vtmp_last_row_ptr_array);
                            check_ptr_array("Vtmp_last_col_ptr_array", batch_count_remain,
                                            Vtmp_last_col_ptr_array);
                            check_ptr_array("Vtmp_last_diag_ptr_array", batch_count_remain,
                                            Vtmp_last_diag_ptr_array);
                        }

                        check_ptr_array("Aj_ptr_array", lbatch_count, Aj_ptr_array);
                        check_ptr_array("Vj_ptr_array", lbatch_count, Vj_ptr_array);

                        check_ptr_array("A_row_ptr_array", lbatch_count, A_row_ptr_array);
                        check_ptr_array("A_col_ptr_array", lbatch_count, A_col_ptr_array);
                        check_ptr_array("A_diag_ptr_array", lbatch_count, A_diag_ptr_array);

                        check_ptr_array("A_ptr_array", batch_count_remain, A_ptr_array);

                        check_ptr_array("Atmp_row_ptr_array", lbatch_count, Atmp_row_ptr_array);
                        check_ptr_array("Atmp_col_ptr_array", lbatch_count, Atmp_col_ptr_array);
                        check_ptr_array("Atmp_diag_ptr_array", lbatch_count, Atmp_diag_ptr_array);

                        check_ptr_array("Atmp_ptr_array", batch_count_remain, Atmp_ptr_array);

                        if(need_V)
                        {
                            check_ptr_array("Vtmp_row_ptr_array", lbatch_count, Vtmp_row_ptr_array);
                            check_ptr_array("Vtmp_col_ptr_array", lbatch_count, Vtmp_col_ptr_array);
                            check_ptr_array("Vtmp_diag_ptr_array", lbatch_count, Vtmp_diag_ptr_array);

                            check_ptr_array("Vtmp_ptr_array", batch_count_remain, Vtmp_ptr_array);
                        }
                    }
#endif
                }

                {
                    // ------------------------
                    // reorder and copy blocks to Atmp,
                    // and to Vtmp if needed
                    // ------------------------
                    I const* const col_map_schedule = d_schedule_large + iround * (even_nblocks);

                    bool const use_schedule = false;

                    if(!use_schedule)
                    {
                        I const ldgmat = nblocks;
                        Istride const strideGmat = (nblocks * nblocks);

                        // clang-format off
                        ROCSOLVER_LAUNCH_KERNEL((symmetrize_matrix_kernel<S, I, S*, Istride>),
					dim3(1, 1, nbz), dim3(nx, 1, 1), 0, stream, 
					c_uplo, n, 
					Gmat, shift_zero, ldgmat, strideGmat,
					batch_count_remain);
                        // clang-format on

                        if(idebug >= 1)
                        {
                            bool const include_diagonal = false;
                            update_norm(include_diagonal, residual);
                            printf("=== before pgreedy_mwm === \n");
                            print_Gmat();
                        }

                        pgreedy_mwm(nblocks, Gmat, mate_array, batch_count_remain, stream);

                        if(idebug >= 1)
                        {
                            print_mate();
                        }
                    }

                    I const* const col_map_mwm = mate_array;
                    Istride const stride_map = (use_schedule) ? 0 : (nblocks);

                    I const* const col_map = (use_schedule) ? col_map_schedule : col_map_mwm;
                    I const* const row_map = col_map;

                    auto const max_thread_blocks = 64 * 1000;
                    auto const nx = NX_THREADS;
                    auto const ny = RSYEVJ_BDIM / nx;

                    auto const lbatch_count = (nblocks_half - 1) * batch_count_remain;
                    auto const nbx = std::min(max_thread_blocks, ceil(n, nx));
                    auto const nby = std::min(max_thread_blocks, ceil(n, ny));
                    auto const nbz = std::min(max_thread_blocks, batch_count_remain);
                    auto const nbz2 = std::min(max_thread_blocks, lbatch_count);

                    {
                        char const c_direction = 'F'; // forward direction

                        if(need_V)
                        {
                            // ------------------------------------
                            // matrix V need only column reordering
                            // ------------------------------------
                            I const* const null_row_map = nullptr;

                            // clang-format off
                            ROCSOLVER_LAUNCH_KERNEL(
                                (reorder_kernel<T, I, Istride>), 
				dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream, 
				c_direction, n, nb, 
				null_row_map, col_map, stride_map, 
				Vtmp_ptr_array, shift_Vtmp, ldvtmp, lstride_Vtmp,
                                Atmp_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
				batch_count_remain);
                            // clang-format on

                            SWAP_Atmp_Vtmp();
                        }

                        // clang-format off
                        ROCSOLVER_LAUNCH_KERNEL((reorder_kernel<T, I, Istride>),
					dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream,
					c_direction, n, nb, 
					row_map, col_map, stride_map,
					A_ptr_array, shift_zero, lda, strideA, 
					Atmp_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
					batch_count_remain);
                        // clang-format on
                    }

                    TRACE(2);
                    // ------------------------------------------------------
                    // prepare to perform Jacobi iteration on independent sets of blocks
                    // ------------------------------------------------------

                    {
                        {
                            // --------------------------
                            // copy diagonal blocks to Aj
                            // --------------------------

                            I const m1 = (2 * nb);
                            I const n1 = (2 * nb);
                            char const cl_uplo = 'A';

                            Istride const strideA_diag = (2 * nb) * (2 * nb);

                            // clang-format off
                            ROCSOLVER_LAUNCH_KERNEL((lacpy_kernel<T, I, T**, T*, Istride>),
				    dim3(nbx, nby, nbz2), dim3(nx, ny, 1), 0, stream, 
				    cl_uplo, m1, n1, 
				    Atmp_diag_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
				    Aj, shift_Aj, ldaj, lstride_Aj, 
				    lbatch_count);
                            // clang-format on

                            TRACE(2);

                            I const m2 = (nb + nb_last);
                            I const n2 = (nb + nb_last);

                            // clang-format off
                            ROCSOLVER_LAUNCH_KERNEL((lacpy_kernel<T, I, T**, T*, Istride>),
					    dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream,
					    cl_uplo, m2, n2, 
					    Atmp_last_diag_ptr_array, shift_Atmp, ldatmp, lstride_Atmp,
					    Aj_last, shift_Aj, ldaj_last, lstride_Aj_last, 
					    batch_count_remain);
                            // clang-format on
                        }

                        TRACE(2);
                        {
                            // -------------------------------------------------------
                            // prepare to perform Jacobi iteration on small diagonal blocks in Aj
                            // -------------------------------------------------------

                            // ----------------------------
                            // set Vj to be diagonal matrix
                            // to store the matrix of eigen vectors
                            // ----------------------------

                            TRACE(2);

                            bool const set_Vj_identity = false;
                            if(set_Vj_identity)
                            {
                                char const c_uplo = 'A';
                                I const m1 = (2 * nb);
                                I const n1 = (2 * nb);

                                T alpha_offdiag = 0;
                                T beta_diag = 1;

                                auto const lbatch_count = (nblocks_half - 1) * batch_count_remain;

                                // clang-format off
                                ROCSOLVER_LAUNCH_KERNEL(
                                    (laset_kernel<T, I, T*, Istride>), 
				    dim3(nbx, nby, nbz2), dim3(nx, ny, 1), 0, stream, 
				    c_uplo, m1, n1, 
				    alpha_offdiag, beta_diag, 
				    Vj, shift_Vj, ldvj, lstride_Vj, 
				    lbatch_count);
                                // clang-format on

                                I const m2 = nb + nb_last;
                                I const n2 = nb + nb_last;

                                // clang-format off
                                ROCSOLVER_LAUNCH_KERNEL((laset_kernel<T, I, T*, Istride>),
						dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream, 
						c_uplo, m2, n2, 
						alpha_offdiag, beta_diag, 
						Vj_last, shift_Vj_last, ldvj_last, lstride_Vj_last, 
						batch_count_remain);
                                // clang-format on
                            }

                            TRACE(2);

                            bool const do_overwrite_A_with_V = true;
                            {
                                // -----------------------------------------
                                // setup options to
                                // preserve the nearly diagonal matrix in Aj
                                // -----------------------------------------
                                rocblas_esort const rsyevj_esort = rocblas_esort_none;
                                size_t const lmemsize = get_lds();

                                // ---------------------------
                                // no need for too many sweeps
                                // since the blocks will be over-written
                                // ---------------------------
                                I const rsyevj_max_sweeps = std::min(max_sweeps, 30);
                                auto const rsyevj_atol = atol / nblocks;

                                // -----------------------------------------
                                // need to store the matrix of eigen vectors
                                // -----------------------------------------
                                rocblas_evect const rsyevj_evect = rocblas_evect_original;
                                rocblas_fill const rsyevj_fill = rocblas_fill_full;

                                I const n1 = (2 * nb);
                                Istride const strideW_Aj = (2 * nb);

                                auto const lbatch_count = (nblocks_half - 1) * batch_count_remain;

                                // clang-format off
                                ROCSOLVER_LAUNCH_KERNEL(
                                    (rsyevj_small_kernel<T, I, S, T**, Istride>), 
				    dim3(1, 1, nbz2), dim3(nx, ny, 1), lmemsize, stream, 
				    rsyevj_esort, rsyevj_evect, rsyevj_fill, 
				    n1, 
				    Aj_ptr_array, shift_Aj, ldaj, lstride_Aj,
                                    rsyevj_atol, eps, residual_Aj, rsyevj_max_sweeps, 
				    n_sweeps_Aj, W_Aj, strideW_Aj, info_Aj, 
				    Vj, lstride_Vj, 
				    lbatch_count, d_schedule_small, do_overwrite_A_with_V);
                                // clang-format on

                                I const n2 = nb + nb_last;

                                Istride const strideW_Aj_last = (nb + nb_last);

                                TRACE(1);
                                // clang-format off
                                ROCSOLVER_LAUNCH_KERNEL(
                                    (rsyevj_small_kernel<T, I, S, T**, Istride>), 
				    dim3(1, 1, nbz), dim3(nx, ny, 1), lmemsize, stream, 
				    rsyevj_esort, rsyevj_evect, rsyevj_fill, 
				    n2, 
				    Aj_last_ptr_array, shift_Aj_last, ldaj_last, lstride_Aj_last, 
				    rsyevj_atol, eps, residual_Aj_last,
                                    rsyevj_max_sweeps, n_sweeps_Aj_last, 
				    W_Aj_last, strideW_Aj_last,
                                    info_Aj_last, 
				    Vj_last, lstride_Vj_last, 
				    batch_count_remain, d_schedule_last, do_overwrite_A_with_V);
                                // clang-format on
                                TRACE(1);

                                if(do_overwrite_A_with_V)
                                {
                                    bool constexpr use_swap_aj_vj = false;
                                    if(use_swap_aj_vj)
                                    {
                                        SWAP_AJ_VJ();
                                    }
                                    else
                                    {
                                        HIP_CHECK(hipMemcpyAsync(Vj, Aj, size_Vj_bytes,
                                                                 hipMemcpyDeviceToDevice, stream));
                                        HIP_CHECK(hipMemcpyAsync(Vj_last, Aj_last, size_Vj_last_bytes,
                                                                 hipMemcpyDeviceToDevice, stream));
                                        HIP_CHECK(hipStreamSynchronize(stream));

                                        assert(size_Vj_bytes == size_Aj_bytes);
                                        assert(size_Vj_last_bytes == size_Aj_last_bytes);
                                    }
                                }
#ifdef NDEBUG
#else
                                if(idebug >= 1)
                                {
                                    S const atol = 1e-5;
                                    auto check_V = [=](I const nb2, std::vector<T>& h_Vj,
                                                       I const ldvj, Istride const lstride_Vj,
                                                       auto const atol, I const lbatch_count,
                                                       bool& isok_Vj) {
                                        isok_Vj = true;

                                        auto eye = [=](auto i, auto j) -> T {
                                            return ((i == j) ? T(1) : T(0));
                                        };

                                        for(auto bid = 0; bid < lbatch_count; bid++)
                                        {
                                            T const* const Vmat_p = &(h_Vj[bid * lstride_Vj]);

                                            auto Vmat = [=](auto i, auto j) -> T {
                                                return (Vmat_p[i + j * ldvj]);
                                            };
                                            // ----------------------------
                                            // compute  norm( eye - Vj'*Vj )
                                            // ----------------------------
                                            for(auto j = 0; j < nb2; j++)
                                            {
                                                for(auto i = 0; i < nb2; i++)
                                                {
                                                    T e_ij = 0;
                                                    for(auto k = 0; k < nb2; k++)
                                                    {
                                                        T const v_kj = Vmat(k, j);
                                                        T const v_ki = Vmat(k, i);
                                                        e_ij += conj(v_ki) * v_kj;
                                                    }
                                                    double const err = std::abs(e_ij - eye(i, j));
                                                    isok_Vj = (err <= atol);
                                                    if(!isok_Vj)
                                                    {
                                                        printf("i=%d,j=%d,bid=%d,e_ij=(%le,%le)\n",
                                                               i, j, bid, std::real(e_ij),
                                                               std::imag(e_ij));
                                                        fflush(stdout);
                                                        return;
                                                    };
                                                }
                                            }
                                        }
                                    };

                                    auto print_V = [=](auto nb2, auto& h_Vj, auto ldvj,
                                                       auto lstride_Vj, auto lbatch_count) {
                                        for(auto bid = 0; bid < lbatch_count; bid++)
                                        {
                                            auto const Vmat_p = &(h_Vj[bid * lstride_Vj]);
                                            auto Vmat = [=](auto i, auto j) {
                                                return (Vmat_p[i + j * ldvj]);
                                            };

                                            for(auto j = 0; j < nb2; j++)
                                            {
                                                for(auto i = 0; i < nb2; i++)
                                                {
                                                    auto const vij = Vmat(i, j);
                                                    printf("Vmat(%d,%d,%d) = (%le,%le)\n", i, j,
                                                           bid, std::real(vij), std::imag(vij));
                                                }
                                            }
                                        }
                                        fflush(stdout);
                                    };

                                    bool isok_Vj = false;
                                    I const n1 = (2 * nb);

                                    auto const len_Vj = size_Vj_bytes / sizeof(T);
                                    std::vector<T> h_Vj(len_Vj);

                                    HIP_CHECK(hipMemcpyAsync((void*)&(h_Vj[0]), Vj, size_Vj_bytes,
                                                             hipMemcpyDeviceToHost, stream));

                                    HIP_CHECK(hipStreamSynchronize(stream));

                                    check_V(n1, h_Vj, ldvj, lstride_Vj, atol, lbatch_count, isok_Vj);

                                    if(!isok_Vj)
                                    {
                                        printf("===== h_Vj ====== \n");
                                        print_V(n1, h_Vj, ldvj, lstride_Vj, lbatch_count);
                                    }
                                    assert(isok_Vj);

                                    I const n2 = nb + nb_last;
                                    bool isok_Vj_last = false;

                                    auto const len_Vj_last = size_Vj_last_bytes / sizeof(T);
                                    std::vector<T> h_Vj_last(len_Vj_last);

                                    HIP_CHECK(hipMemcpyAsync((void*)&(h_Vj_last[0]), Vj_last,
                                                             size_Vj_last_bytes,
                                                             hipMemcpyDeviceToHost, stream));

                                    HIP_CHECK(hipStreamSynchronize(stream));

                                    check_V(n2, h_Vj_last, ldvj_last, lstride_Vj_last, atol,
                                            batch_count_remain, isok_Vj_last);

                                    if(!isok_Vj_last)
                                    {
                                        printf("==== h_Vj_last ===== \n");
                                        print_V(n2, h_Vj_last, ldvj_last, lstride_Vj_last,
                                                batch_count_remain);
                                    }

                                    assert(isok_Vj_last);
                                }
#endif
                                TRACE(1);
                            }
                        }
                    }

                    TRACE(1);
                    {
                        // -----------------------------------------------------
                        // launch batch list to perform Vj' to update block rows
                        //
                        // A <- Vj'*Atmp
                        // -----------------------------------------------------

                        T alpha = 1;
                        T beta = 0;
                        auto const m1 = 2 * nb;
                        auto const n1 = n;
                        auto const k1 = 2 * nb;

                        rocblas_operation const transA = rocblas_operation_conjugate_transpose;
                        rocblas_operation const transB = rocblas_operation_none;

                        auto const lbatch_count = (nblocks_half - 1) * batch_count_remain;
                        TRACE(2);

                        // clang-format off
                        ROCBLAS_CHECK(rocblasCall_gemm(
                            handle, transA, transB, m1, n1, k1, 
			    &alpha, 
			    Vj, shift_Vj, ldvj, lstride_Vj, 
			    Atmp_row_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
			    &beta,
                            A_row_ptr_array, shift_zero, lda, strideA, 
			    lbatch_count, work_rocblas));
                        // clang-format on

                        TRACE(2);

                        // ----------------------------------------------------------
                        // launch batch list to perform Vj' to update last block rows
                        // ----------------------------------------------------------
                        auto const m2 = nb + nb_last;
                        auto const n2 = n;
                        auto const k2 = nb + nb_last;

                        // clang-format off
                        ROCBLAS_CHECK(rocblasCall_gemm(
                            handle, transA, transB, m2, n2, k2, 
			    &alpha, 
			    Vj_last, shift_Vj_last, ldvj_last, lstride_Vj_last, 
			    Atmp_last_row_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
			    &beta, 
			    A_last_row_ptr_array, shift_zero, lda, strideA,
                            batch_count_remain, work_rocblas));
                        // clang-format on
                    }
                    TRACE(1);

                    {
                        // -------------------------------------------------------
                        // launch batch list to perform Vj to update block columns
                        //
                        // Atmp <-  A * Vj
                        // -------------------------------------------------------

                        T alpha = 1;
                        T beta = 0;
                        I const m1 = n;
                        I const n1 = 2 * nb;
                        I const k1 = 2 * nb;

                        rocblas_operation const transA = rocblas_operation_none;
                        rocblas_operation const transB = rocblas_operation_none;

                        auto const lbatch_count = (nblocks_half - 1) * batch_count_remain;

                        TRACE(2);
                        // clang-format off
                        ROCBLAS_CHECK(rocblasCall_gemm(
					handle, transA, transB, m1, n1, k1, 
					&alpha,
				        Vj, shift_Vj, ldvj, lstride_Vj,
				        A_col_ptr_array, shift_zero, lda, strideA, 
					&beta,
				        Atmp_col_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
					lbatch_count, work_rocblas));
                        // clang-format on

                        // -----------------------------------------------------------
                        // launch batch list to perform Vj to update last block column
                        // -----------------------------------------------------------

                        I const m2 = n;
                        I const n2 = nb + nb_last;
                        I const k2 = nb + nb_last;

                        TRACE(2);

                        // clang-format off
                        ROCBLAS_CHECK(rocblasCall_gemm(
                            handle, transA, transB, m2, n2, k2, 
			    &alpha, 
			    Vj_last, shift_Vj_last, ldvj_last, lstride_Vj_last, 
			    A_last_col_ptr_array, shift_zero, lda, strideA, 
			    &beta, 
			    Atmp_last_col_ptr_array, shift_Atmp, ldatmp, lstride_Atmp,
                            batch_count_remain, work_rocblas));
                        // clang-format on
                    }

                    TRACE(1);

                    bool const use_restore_diagonal_blocks = false;
                    if(use_restore_diagonal_blocks)
                    {
                        // ----------------------------------------
                        // copy the diagonal blocks from Aj back to Atmp
                        // ----------------------------------------

                        I const m1 = (2 * nb);
                        I const n1 = (2 * nb);
                        char const cl_uplo = 'A';

                        I const lbatch_count = (nblocks_half - 1) * batch_count_remain;

                        // clang-format off
                        ROCSOLVER_LAUNCH_KERNEL((lacpy_kernel<T, I, T*, T**, Istride>),
					dim3(nbx, nby, nbz2), dim3(nx, ny, 1), 0, stream,
					cl_uplo, m1, n1, 
					Aj, shift_Aj, ldaj, lstride_Aj,
					Atmp_diag_ptr_array, shift_Atmp, ldatmp, lstride_Atmp,
					lbatch_count);
                        // clang-format on

                        TRACE(2);

                        I const m2 = (nb + nb_last);
                        I const n2 = (nb + nb_last);

                        // clang-format off
                        ROCSOLVER_LAUNCH_KERNEL((lacpy_kernel<T, I, T*, T**, Istride>),
					dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream,
					cl_uplo, m2, n2, 
					Aj_last, shift_Aj_last, ldaj_last, lstride_Aj_last, 
					Atmp_last_diag_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
					batch_count_remain);
                        // clang-format on
                    }

                    TRACE(2);
                    bool const use_backward_reorder = use_schedule && (!use_adjust_schedule_large);
                    {
                        // -------------------
                        // copy Atmp back to A
                        // and perhaps undo reordering while copying
                        // -----------------------------

                        if(use_backward_reorder)
                        {
                            char const c_direction = 'B';

                            I const* const col_map = col_map_schedule;
                            I const* const row_map = col_map;
                            Istride const stride_map = 0;
                            // clang-format off
                            ROCSOLVER_LAUNCH_KERNEL(
                                (reorder_kernel<T, I, Istride>), 
				dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream, 
				c_direction, n, nb,
                                row_map, col_map, stride_map,
                                Atmp_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
				A_ptr_array, shift_zero, lda, strideA, 
				batch_count_remain);
                            // clang-format on
                        }
                        else
                        {
                            I const mm = n;
                            I const nn = n;
                            char const cl_uplo = 'A';

                            // clang-format off
                            ROCSOLVER_LAUNCH_KERNEL((lacpy_kernel<T, I, T**, T**, Istride>),
                                                    dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream,
                                                    cl_uplo, mm, nn, 
						    Atmp_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
						    A_ptr_array, shift_zero, lda, strideA, 
						    batch_count_remain);
                            // clang-format on
                        }
                    }

                    TRACE(2);

                    {
                        bool const include_diagonal = false;
                        update_norm(include_diagonal, residual);

                        if(idebug >= 1)
                        {
                            TRACE(1);
                            print_residual();
                        }
                    }

                    if(need_V)
                    {
                        // -------------------------------------------------------
                        // launch batch list to perform Vj to update block columns
                        // for eigen vectors
                        //
                        // Atmp <- Vtmp * Vj
                        // -------------------------------------------------------

                        T alpha = 1;
                        T beta = 0;

                        I const m1 = n;
                        I const n1 = (2 * nb);
                        I const k1 = (2 * nb);

                        auto const lbatch = (nblocks_half - 1) * batch_count_remain;

                        rocblas_operation const transA = rocblas_operation_none;
                        rocblas_operation const transB = rocblas_operation_none;

                        // clang-format off
                        ROCBLAS_CHECK(rocblasCall_gemm(
                            handle, transA, transB, m1, n1, k1,
                            &alpha, 
			    Vtmp_col_ptr_array, shift_Vtmp, ldvtmp, lstride_Vtmp, 
			    Vj, shift_Vj, ldvj, lstride_Vj, 
			    &beta, 
			    Atmp_col_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
			    lbatch_count, work_rocblas));
                        // clang-format on

                        // ------------------------------------------------------------
                        // launch batch list to perform Vj to update last block columns
                        // for eigen vectors
                        // ------------------------------------------------------------
                        I const m2 = n;
                        I const n2 = nb + nb_last;
                        I const k2 = nb + nb_last;

                        // clang-format off
                        ROCBLAS_CHECK(rocblasCall_gemm(
                            handle, transA, transB, m2, n2, k2,
                            &alpha, 
			    Vtmp_last_col_ptr_array, shift_Vtmp, ldvtmp, lstride_Vtmp, 
			    Vj_last, shift_Vj_last, ldvj_last, lstride_Vj_last, 
			    &beta, 
			    Atmp_last_col_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
			    batch_count_remain, work_rocblas));
                        // clang-format on

                        SWAP_Atmp_Vtmp();

                        if(use_backward_reorder)
                        {
                            char const c_direction = 'B';
                            I* const null_row_map = nullptr;
                            I const* const col_map = col_map_schedule;

                            // clang-format off
                            ROCSOLVER_LAUNCH_KERNEL(
                                (reorder_kernel<T, I, Istride>), 
				dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream, 
				c_direction, n, nb, 
				null_row_map,
                                col_map, 
				stride_map,
                                Vtmp_ptr_array, shift_Vtmp, ldatmp, lstride_Vtmp, 
				Atmp_ptr_array, shift_Atmp, ldatmp, lstride_Atmp, 
				batch_count_remain);
                            // clang-format on

                            SWAP_Atmp_Vtmp();
                        }
                    }
                }
            } // end for iround
        } // end for sweeps

        {
            // ---------------------
            // copy out eigen values
            // ---------------------

            // clang-format off
            ROCSOLVER_LAUNCH_KERNEL((copy_diagonal_kernel<T, I, U, Istride>), 
			    dim3(1, 1, nbz), dim3(nx, 1, 1), 0, stream, 
			    n, A, shiftA, lda, strideA, 
			    W, strideW, 
			    batch_count);
            // clang-format on

#ifdef NDEBUG
#else
            if(idebug >= 1)
            {
                printf("after copy_diagonal_kernel\n");
                print_eig(n, W, strideW, batch_count);
            }
#endif
        }

        {
            // -------------------------------------------
            // check whether eigenvalues need to be sorted
            // -------------------------------------------

            // reuse storage
            Istride const stridemap = sizeof(I) * n;
            if(need_sort)
            {
                ROCSOLVER_LAUNCH_KERNEL((sort_kernel<S, I, Istride>), dim3(1, 1, nbz),
                                        dim3(nx, 1, 1), 0, stream,

                                        n, W, strideW, eig_map, stridemap, batch_count);
            }

#ifdef NDEBUG
#else
            if(idebug >= 1)
            {
                printf("after sort_kernel\n");
                print_eig(n, W, strideW, batch_count);
            }
#endif
            // -----------------------------------------------
            // over-write original matrix A with eigen vectors
            // -----------------------------------------------
            if(need_V)
            {
                I* const row_map = nullptr;
                I* const col_map = (need_sort) ? eig_map : nullptr;
                auto const mm = n;
                auto const nn = n;
                ROCSOLVER_LAUNCH_KERNEL((gather2D_kernel<T, I, Istride, T*, U>),
                                        dim3(nbx, nby, nbz), dim3(nx, ny, 1), 0, stream, mm, nn,
                                        row_map, col_map, Vtmp, shift_Vtmp, ldvtmp, lstride_Vtmp, A,
                                        shiftA, lda, strideA, batch_count);
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
            ROCSOLVER_LAUNCH_KERNEL((cal_Gmat_kernel<T, I, S, Istride, U>), dim3(nbx, nby, nbz),
                                    dim3(nx, ny, 1), 0, stream,

                                    n, nb, A, shiftA, lda, strideA, Gmat, need_diagonal, completed,
                                    batch_count);

            TRACE(2);

            ROCSOLVER_LAUNCH_KERNEL((sum_Gmat_kernel<S, I>), dim3(1, 1, nbz), dim3(nx, ny, 1),
                                    sizeof(S), stream, n, nb, Gmat, residual, completed, batch_count);
            auto const nnx = 64;
            auto const nnb = ceil(batch_count, nnx);
            TRACE(2);
            ROCSOLVER_LAUNCH_KERNEL((set_completed_kernel<S, I, Istride>), dim3(nnb, 1, 1),
                                    dim3(nnx, 1, 1), 0, stream, n, nb, Amat_norm, atol, h_sweeps,
                                    n_sweeps, residual, info, completed, batch_count);
        }
        TRACE(2);
#ifdef NDEBUG
#else
        if(idebug >= 1)
        {
            print_Gmat();

            std::vector<I> h_n_sweeps(batch_count);
            std::vector<I> h_info(batch_count);
            std::vector<S> h_residual(batch_count);
            std::vector<I> h_completed(batch_count + 1);
            std::vector<S> h_W(n * batch_count);

            HIP_CHECK(hipMemcpy(&(h_n_sweeps[0]), n_sweeps, sizeof(I) * batch_count,
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&(h_info[0]), info, sizeof(I) * batch_count, hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&(h_residual[0]), residual, sizeof(S) * batch_count,
                                hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(&(h_completed[0]), completed, sizeof(I) * (batch_count + 1),
                                hipMemcpyDeviceToHost));

            HIP_CHECK(hipMemcpy(&(h_W[0]), W, sizeof(S) * n * batch_count, hipMemcpyDeviceToHost));

            printf("completed[0] = %d\n", (int)h_completed[0]);
            for(I bid = 0; bid < batch_count; bid++)
            {
                printf("n_sweeps[%d] = %d, info[%d] = %d, residual[%d] = %le, completed(%d) = %d\n",
                       (int)bid, (int)h_n_sweeps[bid], (int)bid, (int)h_info[bid], (int)bid,
                       (double)h_residual[bid], (int)bid, (int)h_completed[bid + 1]);

                for(auto i = 0; i < n; i++)
                {
                    printf("W[%d,%d] = %le\n", i, bid, (double)h_W[i + bid * n]);
                }
            }
        }
#endif

    } // end large block

    return (rocblas_status_success);
}

ROCSOLVER_END_NAMESPACE

#undef ALLOC_INIT
#undef ALLOC_VJ
#undef ALLOC_AJ
#undef ALLOC_Atmp
#undef ALLOC_Vtmp

#undef ALLOC_A
#undef ALLOC_RESIDUAL_AJ
#undef ALLOC_INFO_AJ
#undef ALLOC_W_AJ
#undef ALLOC_EIG_MAP

#undef ALLOC_N_SWEEPS_AJ
#undef ALLOC_WORK_ROCBLAS
#undef ALLOC_MATE_ARRAY
#undef ALLOC_GMAT
#undef ALLOC_SCHEDULE

#undef ALLOC_ALL
