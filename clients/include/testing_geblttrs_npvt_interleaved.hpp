/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include <stdint.h>

#include "client_util.hpp"
#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

static int64_t indx2(rocblas_int i1, rocblas_int i2, rocblas_int n1_in, rocblas_int n2)
{
    int64_t const n1 = n1_in;
    assert((0 <= i1) && (i1 < n1));
    assert((0 <= i2) && (i2 < n2));

    return (i1 + i2 * n1);
};

static int64_t indx4(rocblas_int i1,
                     rocblas_int i2,
                     rocblas_int i3,
                     rocblas_int i4,
                     rocblas_int n1_in,
                     rocblas_int n2,
                     rocblas_int n3,
                     rocblas_int n4)
{
    int64_t const n1 = n1_in;

    assert((0 <= i1) && (i1 < n1));
    assert((0 <= i2) && (i2 < n2));
    assert((0 <= i3) && (i3 < n3));
    assert((0 <= i4) && (i4 < n4));

    return (i1 + i2 * n1 + i3 * (n1 * n2) + i4 * (n1 * n2 * n3));
};

template <typename T>
void geblttrs_npvt_interleaved_checkBadArgs(const rocblas_handle handle,
                                            const rocblas_int nb,
                                            const rocblas_int nblocks,
                                            const rocblas_int nrhs,
                                            T dA,
                                            const rocblas_int lda,
                                            T dB,
                                            const rocblas_int ldb,
                                            T dC,
                                            const rocblas_int ldc,
                                            T dX,
                                            const rocblas_int ldx,
                                            const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(nullptr, nb, nblocks, nrhs, dA, lda,
                                                              dB, ldb, dC, ldc, dX, ldx, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, lda,
                                                              dB, ldb, dC, ldc, dX, ldx, -1),
                          rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, (T) nullptr,
                                                              lda, dB, ldb, dC, ldc, dX, ldx, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, lda,
                                                              (T) nullptr, ldb, dC, ldc, dX, ldx, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, lda, dB,
                                                              ldb, (T) nullptr, ldc, dX, ldx, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, lda, dB,
                                                              ldb, dC, ldc, (T) nullptr, ldx, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, 0, nblocks, nrhs, (T) nullptr,
                                                              lda, (T) nullptr, ldb, (T) nullptr,
                                                              ldc, (T) nullptr, ldx, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, 0, nrhs, (T) nullptr, lda,
                                                              (T) nullptr, ldb, (T) nullptr, ldc,
                                                              (T) nullptr, ldx, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, 0, dA, lda, dB,
                                                              ldb, dC, ldc, (T) nullptr, ldx, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA, lda,
                                                              dB, ldb, dC, ldc, dX, ldx, 0),
                          rocblas_status_success);
}

template <typename T>
void testing_geblttrs_npvt_interleaved_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int const nb = 1;
    rocblas_int const nblocks = 2;
    rocblas_int const nrhs = 1;
    rocblas_int const lda = 1;
    rocblas_int const ldb = 1;
    rocblas_int const ldc = 1;
    rocblas_int const ldx = 1;
    rocblas_int const bc = 1;

    // memory allocations
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dB(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    device_strided_batch_vector<T> dX(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dX.memcheck());

    // check bad arguments
    geblttrs_npvt_interleaved_checkBadArgs(handle, nb, nblocks, nrhs, dA.data(), lda, dB.data(),
                                           ldb, dC.data(), ldc, dX.data(), ldx, bc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void geblttrs_npvt_interleaved_initData(const rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
                                        Td& dA,
                                        const rocblas_int lda,
                                        Td& dB,
                                        const rocblas_int ldb,
                                        Td& dC,
                                        const rocblas_int ldc,
                                        Td& dX,
                                        const rocblas_int ldx,
                                        const rocblas_int bc,
                                        Th& hA_,
                                        Th& hB_,
                                        Th& hC_,
                                        Th& hX_,
                                        Th& hRHS_)
{
    if(CPU)
    {
        int info;
        size_t const n = nb * nblocks;
        size_t const ldM = n;
        size_t const ldXX = n;
        size_t const ldXB = n;

        std::vector<T> M_(ldM * n);
        std::vector<T> XX_(ldXX * nrhs);
        std::vector<T> XB_(ldXB * nrhs);

#define M(ii, jj) M_[indx2(ii, jj, ldM, n)]
#define XX(ii, jj) XX_[indx2(ii, jj, ldXX, nrhs)]
#define XB(ii, jj) XB_[indx2(ii, jj, ldXB, nrhs)]

        // std::vector<rocblas_int> ipiv(nb);

        // initialize blocks of the original matrix
        rocblas_init<T>(hA_, true);
        rocblas_init<T>(hB_, false);
        rocblas_init<T>(hC_, false);

        // initialize solution vectors
        rocblas_init<T>(hX_, false);

        rocblas_int const ldrhs = ldx;

#define hA(b, i, j, k) hA_[0][indx4(b, i, j, k, bc, lda, nb, nblocks - 1)]
#define hB(b, i, j, k) hB_[0][indx4(b, i, j, k, bc, ldb, nb, nblocks)]
#define hC(b, i, j, k) hC_[0][indx4(b, i, j, k, bc, ldc, nb, nblocks - 1)]
#define hX(b, i, k, irhs) hX_[0][indx4(b, i, k, irhs, bc, ldx, nblocks, nrhs)]
#define hRHS(b, i, k, irhs) hRHS_[0][indx4(b, i, k, irhs, bc, ldrhs, nblocks, nrhs)]

        // adjust hA_, hB_, hC_ to avoid singularities
        for(rocblas_int k = 0; k < nblocks; k++)
        {
            for(rocblas_int j = 0; j < nb; j++)
            {
                for(rocblas_int i = 0; i < nb; i++)
                {
                    for(rocblas_int b = 0; b < bc; b++)
                    {
                        bool const is_diag = (i == j);
                        if(is_diag)
                        {
                            hB(b, i, j, k) += 400;
                        }
                        else
                        {
                            hB(b, i, j, k) -= 4;
                        };

                        if(k < (nblocks - 1))
                        {
                            hA(b, i, j, k) -= 4;
                            hC(b, i, j, k) -= 4;
                        };
                    };
                };
            };
        };

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // form original matrix M and scale to avoid singularities
            for(rocblas_int j = 0; j < n; j++)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    M(i, j) = 0;
                };
            };

            for(rocblas_int k = 0; k < nblocks; k++)
            {
                // diagonal blocks
                for(rocblas_int j = 0; j < nb; j++)
                {
                    for(rocblas_int i = 0; i < nb; i++)
                    {
                        T const bij = hB(b, i, j, k);
                        auto const ii = indx2(i, k, nb, nblocks);
                        auto const jj = indx2(j, k, nb, nblocks);
                        M(ii, jj) = bij;
                    };
                };
            }; // end for k

            // off-diagonal blocks
            for(rocblas_int k = 0; k < (nblocks - 1); k++)
            {
                for(rocblas_int j = 0; j < nb; j++)
                {
                    for(rocblas_int i = 0; i < nb; i++)
                    {
                        // lower-diagonal block
                        {
                            T const aij = hA(b, i, j, k);

                            auto const ii = indx2(i, k, nb, nblocks) + nb;
                            auto const jj = indx2(j, k, nb, nblocks);

                            M(ii, jj) = aij;
                        };

                        // upper-diagonal block
                        {
                            T const cij = hC(b, i, j, k);

                            auto const ii = indx2(i, k, nb, nblocks);
                            auto const jj = indx2(j, k, nb, nblocks) + nb;

                            M(ii, jj) = cij;
                        };
                    };
                };
            }; // end for k

            // move blocks of X to full matrix XX
            for(rocblas_int irhs = 0; irhs < nrhs; irhs++)
            {
                for(rocblas_int k = 0; k < nblocks; k++)
                {
                    for(rocblas_int i = 0; i < nb; i++)
                    {
                        T const x_ik_irhs = hX(b, i, k, irhs);
                        rocblas_int const ii = indx2(i, k, nb, nblocks);
                        XX(ii, irhs) = x_ik_irhs;
                    };
                };
            };

            // generate the full matrix of right-hand-side vectors XB by computing M * XX
            {
                rocblas_int const mm = n;
                rocblas_int const nn = nrhs;
                rocblas_int const kk = n;
                T const alpha = 1;
                T const beta = 0;

                T* Ap = &(M(0, 0));
                rocblas_int const ld1 = ldM;
                T* Bp = &(XX(0, 0));
                rocblas_int const ld2 = ldXX;
                T* Cp = &(XB(0, 0));
                rocblas_int const ld3 = ldXB;

                cpu_gemm(rocblas_operation_none, rocblas_operation_none, mm, nn, kk, alpha, Ap, ld1,
                         Bp, ld2, beta, Cp, ld3);
            };

            // move XB to block format in hRHS
            for(rocblas_int irhs = 0; irhs < nrhs; irhs++)
            {
                for(rocblas_int k = 0; k < nblocks; k++)
                {
                    for(rocblas_int i = 0; i < nb; i++)
                    {
                        rocblas_int const ii = indx2(i, k, nb, nblocks);
                        T const xb_ik_irhs = XB(ii, irhs);
                        hRHS(b, i, k, irhs) = xb_ik_irhs;
                    };
                };
            };

        }; // end for b
    };

    // now copy data to the GPU
    if(GPU)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA_));
        CHECK_HIP_ERROR(dB.transfer_from(hB_));
        CHECK_HIP_ERROR(dC.transfer_from(hC_));

        // copy hRHS to dX
        CHECK_HIP_ERROR(dX.transfer_from(hRHS_));
    };
}
#undef hA
#undef hB
#undef hC
#undef hX

#undef M
#undef XX
#undef XB

template <typename T, typename Td, typename Th>
void geblttrs_npvt_interleaved_getError(const rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        const rocblas_int nrhs,
                                        Td& dA,
                                        const rocblas_int lda,
                                        Td& dB,
                                        const rocblas_int ldb,
                                        Td& dC,
                                        const rocblas_int ldc,
                                        Td& dX,
                                        const rocblas_int ldx,
                                        const rocblas_int bc,
                                        Th& hA_,
                                        Th& hB_,
                                        Th& hC_,
                                        Th& hX_,
                                        Th& hXRes_,
                                        double* max_err)
{
#define hA(b, i, j, k) hA_[0][indx4(b, i, j, k, bc, lda, nb, nblocks - 1)]
#define hB(b, i, j, k) hB_[0][indx4(b, i, j, k, bc, ldb, nb, nblocks)]
#define hC(b, i, j, k) hC_[0][indx4(b, i, j, k, bc, ldc, nb, nblocks - 1)]
#define hX(b, i, k, irhs) hX_[0][indx4(b, i, k, irhs, bc, ldx, nblocks, nrhs)]
#define hXRes(b, i, k, irhs) hXRes_[0][indx4(b, i, k, irhs, bc, ldx, nblocks, nrhs)]

    size_t const n = nb * nblocks;
    size_t const ldXX = n;
    size_t const ldXXRes = n;

    std::vector<T> XX_(ldXX * nrhs);
    std::vector<T> XXRes_(ldXXRes * nrhs);

#define XX(ii, irhs) XX_[indx2(ii, irhs, ldXX, nrhs)]
#define XXRes(ii, irhs) XXRes_[indx2(ii, irhs, ldXXRes, nrhs)]

    // input data initialization
    geblttrs_npvt_interleaved_initData<true, true, T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC,
                                                      ldc, dX, ldx, bc, hA_, hB_, hC_, hX_, hXRes_);

    // execute computations
    // GPU lapack

    {
        device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
        CHECK_HIP_ERROR(dInfo.memcheck());
        // perform factorization
        CHECK_ROCBLAS_ERROR(rocsolver_geblttrf_npvt_interleaved(
            handle, nb, nblocks, dA.data(), lda, dB.data(), ldb, dC.data(), ldc, dInfo.data(), bc));
    };

    // perform solve
    CHECK_ROCBLAS_ERROR(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA.data(),
                                                            lda, dB.data(), ldb, dC.data(), ldc,
                                                            dX.data(), ldx, bc));
    CHECK_HIP_ERROR(hXRes_.transfer_from(dX));

    // // CPU lapack
    // for(rocblas_int b = 0; b < bc; ++b)
    // {
    //     cpu_getrs(trans, n, nrhs, hA[b], lda, hIpiv[b], hB[b], ldb);
    // }

    double err = 0;
    *max_err = 0;

    // error is ||hX - hXRes|| / ||hX||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    for(rocblas_int b = 0; b < bc; ++b)
    {
        // move blocks of X to full matrix XX
        for(rocblas_int irhs = 0; irhs < nrhs; irhs++)
        {
            for(rocblas_int k = 0; k < nblocks; k++)
            {
                for(rocblas_int i = 0; i < nb; i++)
                {
                    auto const ii = indx2(i, k, nb, nblocks);

                    T const x_ik = hX(b, i, k, irhs);
                    XX(ii, irhs) = x_ik;

                    T const xres_ik = hXRes(b, i, k, irhs);
                    XXRes(ii, irhs) = xres_ik;
                };
            };
        };

        err = norm_error('F', n, nrhs, n, &(XX(0, 0)), &(XXRes(0, 0)));
        *max_err = err > *max_err ? err : *max_err;
    };
}

#undef hA
#undef hB
#undef hC
#undef hX
#undef hXRes
#undef XX
#undef XXRes

template <typename T, typename Td, typename Th>
void geblttrs_npvt_interleaved_getPerfData(const rocblas_handle handle,
                                           const rocblas_int nb,
                                           const rocblas_int nblocks,
                                           const rocblas_int nrhs,
                                           Td& dA,
                                           const rocblas_int lda,
                                           Td& dB,
                                           const rocblas_int ldb,
                                           Td& dC,
                                           const rocblas_int ldc,
                                           Td& dX,
                                           const rocblas_int ldx,
                                           const rocblas_int bc,
                                           Th& hA,
                                           Th& hB,
                                           Th& hC,
                                           Th& hX,
                                           Th& hXRes,
                                           double* gpu_time_used,
                                           double* cpu_time_used,
                                           const rocblas_int hot_calls,
                                           const int profile,
                                           const bool profile_kernels,
                                           const bool perf)
{
    if(!perf)
    {
        // geblttrs_npvt_interleaved_initData<true, false, T>(
        //     handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC, ldc, dX, ldx, bc, hA, hB, hC, hX, hXRes);

        // // cpu-lapack performance (only if not in perf mode)
        // *cpu_time_used = get_time_us_no_sync();
        // for(rocblas_int b = 0; b < bc; ++b)
        // {
        //    cpu_getrs(trans, n, nrhs, hA[b], lda, hIpiv[b], hB[b], ldb);
        // }
        // *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
        *cpu_time_used = nan("");
    }

    geblttrs_npvt_interleaved_initData<true, false, T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb,
                                                       dC, ldc, dX, ldx, bc, hA, hB, hC, hX, hXRes);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geblttrs_npvt_interleaved_initData<false, true, T>(handle, nb, nblocks, nrhs, dA, lda, dB,
                                                           ldb, dC, ldc, dX, ldx, bc, hA, hB, hC,
                                                           hX, hXRes);

        CHECK_ROCBLAS_ERROR(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs,
                                                                dA.data(), lda, dB.data(), ldb,
                                                                dC.data(), ldc, dX.data(), ldx, bc));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        geblttrs_npvt_interleaved_initData<false, true, T>(handle, nb, nblocks, nrhs, dA, lda, dB,
                                                           ldb, dC, ldc, dX, ldx, bc, hA, hB, hC,
                                                           hX, hXRes);

        start = get_time_us_sync(stream);
        rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA.data(), lda, dB.data(),
                                            ldb, dC.data(), ldc, dX.data(), ldx, bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_geblttrs_npvt_interleaved(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int nb = argus.get<rocblas_int>("nb");
    rocblas_int nblocks = argus.get<rocblas_int>("nblocks");
    rocblas_int nrhs = argus.get<rocblas_int>("nrhs");
    rocblas_int lda = argus.get<rocblas_int>("lda", nb);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", nb);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", nb);
    rocblas_int ldx = argus.get<rocblas_int>("ldx", nb);

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t const size_A = max(1, bc * size_t(lda) * nb * (nblocks - 1));
    size_t const size_B = max(1, bc * size_t(ldb) * nb * nblocks);
    size_t const size_C = max(1, bc * size_t(ldc) * nb * (nblocks - 1));
    size_t const size_X = max(1, bc * size_t(ldx) * nblocks * nrhs);
    size_t const size_XRes = size_X;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // check invalid sizes
    bool invalid_size = (nb < 0 || nblocks < 0 || nrhs < 0 || lda < nb || ldb < nb || ldc < nb
                         || ldx < nb || bc < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrs_npvt_interleaved(
                                  handle, nb, nblocks, nrhs, (T*)nullptr, lda, (T*)nullptr, ldb,
                                  (T*)nullptr, ldc, (T*)nullptr, ldx, bc),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, (T*)nullptr,
                                                              lda, (T*)nullptr, ldb, (T*)nullptr,
                                                              ldc, (T*)nullptr, ldx, bc));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // memory allocations
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hB(size_B, 1, size_B, 1);
    host_strided_batch_vector<T> hC(size_C, 1, size_C, 1);
    host_strided_batch_vector<T> hX(size_X, 1, size_X, 1);
    host_strided_batch_vector<T> hXRes(size_XRes, 1, size_XRes, 1);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dB(size_B, 1, size_B, 1);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, 1);
    device_strided_batch_vector<T> dX(size_X, 1, size_X, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_B)
        CHECK_HIP_ERROR(dB.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());
    if(size_X)
        CHECK_HIP_ERROR(dX.memcheck());

    // check quick return
    if(nb == 0 || nblocks == 0 || nrhs == 0 || bc == 0)
    {
        EXPECT_ROCBLAS_STATUS(
            rocsolver_geblttrs_npvt_interleaved(handle, nb, nblocks, nrhs, dA.data(), lda,
                                                dB.data(), ldb, dC.data(), ldc, dX.data(), ldx, bc),
            rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        geblttrs_npvt_interleaved_getError<T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC, ldc,
                                              dX, ldx, bc, hA, hB, hC, hX, hXRes, &max_error);

    // collect performance data
    if(argus.timing)
        geblttrs_npvt_interleaved_getPerfData<T>(handle, nb, nblocks, nrhs, dA, lda, dB, ldb, dC,
                                                 ldc, dX, ldx, bc, hA, hB, hC, hX, hXRes,
                                                 &gpu_time_used, &cpu_time_used, hot_calls,
                                                 argus.profile, argus.profile_kernels, argus.perf);

    // validate results for rocsolver-test
    // using nb * machine_precision as tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, nb);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("nb", "nblocks", "nrhs", "lda", "ldb", "ldc", "ldx", "batch_c");
            rocsolver_bench_output(nb, nblocks, nrhs, lda, ldb, ldc, ldx, bc);
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_GEBLTTRS_NPVT_INTERLEAVED(...) \
    extern template void testing_geblttrs_npvt_interleaved<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEBLTTRS_NPVT_INTERLEAVED, FOREACH_SCALAR_TYPE, APPLY_STAMP)
