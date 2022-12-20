/* ************************************************************************
 * Copyright (c) 2021-2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "client_util.hpp"
#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

#include <cassert>

#ifndef INDX_H
#define INDX_H

static int64_t indx4(int i1, int i2, int i3, int i4, int n1_in, int n2, int n3, int n4)
{
    int64_t const n1 = n1_in;

    assert((0 <= i1) && (i1 < n1));
    assert((0 <= i2) && (i2 < n2));
    assert((0 <= i3) && (i3 < n3));
    assert((0 <= i4) && (i4 < n4));

    return (i1 + i2 * n1 + i3 * (n1 * n2) + i4 * (n1 * n2 * n3));
};

static int64_t indx3(int i1, int i2, int i3, int n1_in, int n2, int n3)
{
    int64_t const n1 = n1_in;

    assert((0 <= i1) && (i1 < n1));
    assert((0 <= i2) && (i2 < n2));
    assert((0 <= i3) && (i3 < n3));

    return (i1 + i2 * n1 + i3 * (n1 * n2));
};

static int64_t indx2(int i1, int i2, int n1_in, int n2)
{
    int64_t const n1 = n1_in;

    assert((0 <= i1) && (i1 < n1));
    assert((0 <= i2) && (i2 < n2));

    return (i1 + i2 * n1);
};

#endif

template <typename T, typename U>
void geblttrf_npvt_interleaved_checkBadArgs(const rocblas_handle handle,
                                            const rocblas_int nb,
                                            const rocblas_int nblocks,
                                            T dA,
                                            const rocblas_int lda,
                                            T dB,
                                            const rocblas_int ldb,
                                            T dC,
                                            const rocblas_int ldc,
                                            U dInfo,
                                            const rocblas_int bc)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(nullptr, nb, nblocks, dA, lda, dB,
                                                              ldb, dC, ldc, dInfo, bc),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, lda, dB, ldb,
                                                              dC, ldc, dInfo, -1),
                          rocblas_status_invalid_size);

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, (T) nullptr, lda,
                                                              dB, ldb, dC, ldc, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, lda,
                                                              (T) nullptr, ldb, dC, ldc, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, lda, dB, ldb,
                                                              (T) nullptr, ldc, dInfo, bc),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, lda, dB, ldb,
                                                              dC, ldc, (U) nullptr, bc),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, 0, nblocks, (T) nullptr, lda,
                                                              (T) nullptr, ldb, (T) nullptr, ldc,
                                                              dInfo, bc),
                          rocblas_status_success);
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, 0, (T) nullptr, lda,
                                                              (T) nullptr, ldb, (T) nullptr, ldc,
                                                              dInfo, bc),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA, lda, dB, ldb,
                                                              dC, ldc, (U) nullptr, 0),
                          rocblas_status_success);
}

template <typename T>
void testing_geblttrf_npvt_interleaved_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int nb = 1;
    rocblas_int nblocks = 2;
    rocblas_int lda = 1;
    rocblas_int ldb = 1;
    rocblas_int ldc = 1;
    rocblas_int bc = 1;

    // memory allocations
    device_strided_batch_vector<T> dA(1, 1, 1, 1);
    device_strided_batch_vector<T> dB(1, 1, 1, 1);
    device_strided_batch_vector<T> dC(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check bad arguments
    geblttrf_npvt_interleaved_checkBadArgs(handle, nb, nblocks, dA.data(), lda, dB.data(), ldb,
                                           dC.data(), ldc, dInfo.data(), bc);
}

template <bool CPU, bool GPU, typename T, typename Td, typename Th>
void geblttrf_npvt_interleaved_initData(const rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        Td& dA,
                                        const rocblas_int lda,
                                        Td& dB,
                                        const rocblas_int ldb,
                                        Td& dC,
                                        const rocblas_int ldc,
                                        const rocblas_int bc,
                                        Th& hA_,
                                        Th& hB_,
                                        Th& hC_,
                                        const bool singular)
{
#define hB(ibatch, i, j, iblock) hB_[0][indx4(ibatch, i, j, iblock, bc, ldb, nb, nblocks)]
#define hA(ibatch, i, j, iblock) hA_[0][indx4(ibatch, i, j, iblock, bc, lda, nb, nblocks - 1)]
#define hC(ibatch, i, j, iblock) hC_[0][indx4(ibatch, i, j, iblock, bc, ldc, nb, nblocks - 1)]

    if(CPU)
    {
        T tmp;
        rocblas_init<T>(hA_, true);
        rocblas_init<T>(hB_, false);
        rocblas_init<T>(hC_, false);

        size_t const n = nb * nblocks;

        for(rocblas_int b = 0; b < bc; ++b)
        {
            // scale to avoid singularities
            // leaving matrix as diagonal dominant so that pivoting is not required
            for(rocblas_int i = 0; i < nb; i++)
            {
                for(rocblas_int j = 0; j < nb; j++)
                {
                    for(rocblas_int k = 0; k < nblocks; k++)
                    {
                        if(i == j)
                        {
                            hB(b, i, j, k) += 400;
                        }
                        else
                        {
                            hB(b, i, j, k) -= 4;
                        };
                    }

                    for(rocblas_int k = 0; k < nblocks - 1; k++)
                    {
                        hA(b, i, j, k) -= 4;
                        hC(b, i, j, k) -= 4;
                    }
                }
            }

            if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
            {
                // When required, add some singularities
                // (always the same elements for debugging purposes)

                rocblas_int jj = n / 4 + b;
                jj -= (jj / n) * n;
                rocblas_int j = jj % nb;
                rocblas_int k = jj / nb;
                for(rocblas_int i = 0; i < nb; i++)
                {
                    // zero the jj-th column
                    // hB[k][b + i * bc + j * bc * ldb] = 0;
                    hB(b, i, j, k) = 0;
                    if(k < nblocks - 1)
                    {
                        // hA[k][b + i * bc + j * bc * lda] = 0;
                        hA(b, i, j, k) = 0;
                    };
                    if(k > 0)
                    {
                        // hC[k - 1][b + i * bc + j * bc * ldc] = 0;
                        hC(b, i, j, k - 1);
                    };
                }

                jj = n / 2 + b;
                jj -= (jj / n) * n;
                j = jj % nb;
                k = jj / nb;
                for(rocblas_int i = 0; i < nb; i++)
                {
                    // zero the jj-th column
                    // hB[k][b + i * bc + j * bc * ldb] = 0;
                    hB(b, i, j, k) = 0;
                    if(k < nblocks - 1)
                    {
                        // hA[k][b + i * bc + j * bc * lda] = 0;
                        hA(b, i, j, k) = 0;
                    };
                    if(k > 0)
                    {
                        // hC[k - 1][b + i * bc + j * bc * ldc] = 0;
                        hC(b, i, j, k - 1) = 0;
                    };
                }

                jj = n - 1 + b;
                jj -= (jj / n) * n;
                j = jj % nb;
                k = jj / nb;
                for(rocblas_int i = 0; i < nb; i++)
                {
                    // zero the jj-th column
                    // hB[k][b + i * bc + j * bc * ldb] = 0;
                    hB(b, i, j, k) = 0;
                    if(k < nblocks - 1)
                    {
                        // hA[k][b + i * bc + j * bc * lda] = 0;
                        hA(b, i, j, k) = 0;
                    };
                    if(k > 0)
                    {
                        // hC[k - 1][b + i * bc + j * bc * ldc] = 0;
                        hC(b, i, j, k - 1) = 0;
                    };
                }
            }
        }
    }

    // now copy data to the GPU
    if(GPU)
    {
        CHECK_HIP_ERROR(dA.transfer_from(hA_));
        CHECK_HIP_ERROR(dB.transfer_from(hB_));
        CHECK_HIP_ERROR(dC.transfer_from(hC_));
    }
}
#undef hA
#undef hB
#undef hC

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void geblttrf_npvt_interleaved_getError(const rocblas_handle handle,
                                        const rocblas_int nb,
                                        const rocblas_int nblocks,
                                        Td& dA,
                                        const rocblas_int lda,
                                        Td& dB,
                                        const rocblas_int ldb,
                                        Td& dC,
                                        const rocblas_int ldc,
                                        Ud& dInfo,
                                        const rocblas_int bc,
                                        Th& hA_,
                                        Th& hB_,
                                        Th& hBRes_,
                                        Th& hC_,
                                        Th& hCRes_,
                                        Uh& hInfo,
                                        Uh& hInfoRes,
                                        double* max_err,
                                        const bool singular)
{
    // -----------------------------------------
    // set idebug = 1 to turn on debug messages
    // -----------------------------------------
    int constexpr idebug = 0;

#define hB(ibatch, i, j, iblock) hB_[0][indx4(ibatch, i, j, iblock, bc, ldb, nb, nblocks)]
#define hA(ibatch, i, j, iblock) hA_[0][indx4(ibatch, i, j, iblock, bc, lda, nb, nblocks - 1)]
#define hC(ibatch, i, j, iblock) hC_[0][indx4(ibatch, i, j, iblock, bc, ldc, nb, nblocks - 1)]

#define hBRes(b, i, j, k) hBRes_[0][indx4(b, i, j, k, bc, ldb, nb, nblocks)]
#define hCRes(b, i, j, k) hCRes_[0][indx4(b, i, j, k, bc, ldc, nb, nblocks)]
    size_t const n = nb * nblocks;

    size_t const ldLk = nb;
    size_t const ldUk = nb;
    size_t const ldDk = nb;
    std::vector<T> Lk_(ldLk * nb);
    std::vector<T> Uk_(ldUk * nb);
    std::vector<T> Dk_(ldDk * nb);

#define Lk(i, j) Lk_[indx2(i, j, ldLk, nb)]
#define Uk(i, j) Uk_[indx2(i, j, ldUk, nb)]
#define Dk(i, j) Dk_[indx2(i, j, ldDk, nb)]

    rocblas_int const ldL = n;
    rocblas_int const ldU = n;
    rocblas_int const ldM = n;
    rocblas_int const ldMRes = n;

    std::vector<T> L_(ldL * n);
    std::vector<T> U_(ldU * n);
    std::vector<T> M_(ldM * n);
    std::vector<T> MRes_(ldMRes * n);

#define L(i, j) L_[indx2(i, j, ldL, n)]
#define U(i, j) U_[indx2(i, j, ldU, n)]
#define M(i, j) M_[indx2(i, j, ldM, n)]
#define MRes(i, j) MRes_[indx2(i, j, ldM, n)]

    // input data initialization
    geblttrf_npvt_interleaved_initData<true, true, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC,
                                                      ldc, bc, hA_, hB_, hC_, singular);

    auto print_mat = [=](std::string name, int nb, int nblocks, T* Bp, int ldb, int batch_count) {
        std::cout << name << " = zeros([" << batch_count << "," << ldb << "," << nb << ","
                  << nblocks << "]);\n";

        for(auto k = 0; k < nblocks; k++)
        {
            for(auto j = 0; j < nb; j++)
            {
                for(auto i = 0; i < nb; i++)
                {
                    for(auto ibatch = 0; ibatch < batch_count; ibatch++)
                    {
                        T const bval = Bp[ibatch + i * batch_count + j * (batch_count * ldb)
                                          + k * (batch_count * ldb * nb)];

                        std::cout << name << "(" << ibatch + 1 << "," << i + 1 << "," << j + 1
                                  << "," << k + 1 << ") = " << bval << ";\n";
                    };
                };
            };
        };
    };

    auto print_fullmat = [=](std::string name, int m, int n, T* A, int lda) {
        std::cout << name << "= zeros([" << m << "," << n << "]);\n";
        for(auto j = 0; j < n; j++)
        {
            for(auto i = 0; i < m; i++)
            {
                T aij = A[i + j * lda];
                if(std::abs(aij) > 0)
                {
                    std::cout << name << "(" << i + 1 << "," << j + 1 << ") = " << aij << ";\n";
                };
            };
        };
    };

    // execute computations
    // GPU lapack

    CHECK_ROCBLAS_ERROR(rocsolver_geblttrf_npvt_interleaved(
        handle, nb, nblocks, dA.data(), lda, dB.data(), ldb, dC.data(), ldc, dInfo.data(), bc));
    CHECK_HIP_ERROR(hBRes_.transfer_from(dB));
    CHECK_HIP_ERROR(hCRes_.transfer_from(dC));
    CHECK_HIP_ERROR(hInfoRes.transfer_from(dInfo));

    // // CPU lapack
    // for(rocblas_int b = 0; b < bc; ++b)
    // {
    //     cpu_getrf(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
    // }

    // check info for singularities
    double err = 0;
    *max_err = 0;
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(singular && (b == bc / 4 || b == bc / 2 || b == bc - 1))
        {
            if(hInfoRes[b][0] <= 0)
            {
                err++;
                printf("bc=%d, singular=%d, b=%d, hInfoRes[b][0] = %d <= 0\n", bc, singular, b,
                       hInfoRes[b][0]);
                printf("nb=%d, nblocks=%d\n", nb, nblocks);
            };
        }
        else
        {
            if(hInfoRes[b][0] != 0)
            {
                err++;
                printf("bc=%d, singular=%d, b=%d, hInfoRes[b][0] = %d != 0\n", bc, singular, b,
                       hInfoRes[b][0]);
                printf("nb=%d, nblocks=%d\n", nb, nblocks);
            };
        }
    }
    *max_err += err;

    // error is ||M - MRes|| / ||M||
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY ISSUES.
    // IT MIGHT BE REVISITED IN THE FUTURE)
    // using frobenius norm
    for(rocblas_int b = 0; b < bc; ++b)
    {
        if(hInfoRes[b][0] == 0)
        {
            for(rocblas_int j = 0; j < n; j++)
            {
                for(rocblas_int i = 0; i < n; i++)
                {
                    L(i, j) = 0;
                };
            };

            // compute original blocks B and store in L
            for(rocblas_int k = 0; k < nblocks; k++)
            {
                // split diagonal block into Lk, Uk
                for(rocblas_int j = 0; j < nb; j++)
                {
                    for(rocblas_int i = 0; i < nb; i++)
                    {
                        bool const is_lower = (i > j);
                        bool const is_upper = (i < j);
                        bool const is_diag = (i == j);

                        T const dij = hBRes(b, i, j, k);

                        Lk(i, j) = (is_lower) ? dij : (is_diag) ? 1 : 0;

                        Uk(i, j) = (is_upper || is_diag) ? dij : 0;

                        Dk(i, j) = 0;
                    };
                };

                // compute Dk  = Lk * Uk
                {
                    rocblas_int const mm = nb;
                    rocblas_int const nn = nb;
                    rocblas_int const kk = nb;
                    T const alpha = 1;
                    T const beta = 0;
                    T* Ap = &(Lk(0, 0));
                    rocblas_int const ld1 = ldLk;
                    T* Bp = &(Uk(0, 0));
                    rocblas_int const ld2 = ldUk;
                    T* Cp = &(Dk(0, 0));
                    rocblas_int const ld3 = ldDk;

                    cpu_gemm(rocblas_operation_none, rocblas_operation_none, mm, nn, kk, alpha, Ap,
                             ld1, Bp, ld2, beta, Cp, ld3);
                };
                // copy Dk into diagonal blocks of L
                {
                    for(rocblas_int j = 0; j < nb; j++)
                    {
                        for(rocblas_int i = 0; i < nb; i++)
                        {
                            T const dij = Dk(i, j);

                            auto const ii = indx2(i, k, nb, nblocks);
                            auto const jj = indx2(j, k, nb, nblocks);
                            L(ii, jj) = dij;
                        };
                    };
                };

            }; // end for k

            // initialize U as identity matrix
            for(rocblas_int jj = 0; jj < n; jj++)
            {
                for(rocblas_int ii = 0; ii < n; ii++)
                {
                    bool const is_diag = (ii == jj);
                    U(ii, jj) = (is_diag) ? 1 : 0;
                };
            };

            // move blocks A and CRes  into full matrices L and U
            for(rocblas_int k = 0; k < (nblocks - 1); k++)
            {
                for(rocblas_int j = 0; j < nb; j++)
                {
                    for(rocblas_int i = 0; i < nb; i++)
                    {
                        // lower diagonal block
                        {
                            T const aij = hA(b, i, j, k);
                            auto const ii = indx2(i, k, nb, nblocks) + nb;
                            auto const jj = indx2(j, k, nb, nblocks);
                            L(ii, jj) = aij;
                        };

                        // upper diagonal block
                        {
                            T const cij = hCRes(b, i, j, k);
                            auto const ii = indx2(i, k, nb, nblocks);
                            auto const jj = indx2(j, k, nb, nblocks) + nb;
                            U(ii, jj) = cij;
                        };
                    };
                };
            }; // end for k

            // compute original matrix and store in MRes
            {
                rocblas_int const mm = n;
                rocblas_int const nn = n;
                rocblas_int const kk = n;
                T const alpha = 1;
                T const beta = 0;
                T* Ap = &(L(0, 0));
                rocblas_int const ld1 = ldL;
                T* Bp = &(U(0, 0));
                rocblas_int const ld2 = ldU;
                T* Cp = &(MRes(0, 0));
                rocblas_int const ld3 = ldMRes;

                cpu_gemm(rocblas_operation_none, rocblas_operation_none, mm, nn, kk, alpha, Ap, ld1,
                         Bp, ld2, beta, Cp, ld3);
            };

            // form original matrix from original blocks
            for(auto j = 0; j < n; j++)
            {
                for(auto i = 0; i < n; i++)
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
                if(k < (nblocks - 1))
                {
                    for(rocblas_int j = 0; j < nb; j++)
                    {
                        for(rocblas_int i = 0; i < nb; i++)
                        {
                            {
                                // sub-diagonal block
                                T const aij = hA(b, i, j, k);

                                auto const ii = indx2(i, k, nb, nblocks) + nb;
                                auto const jj = indx2(j, k, nb, nblocks);
                                M(ii, jj) = aij;
                            };

                            {
                                // super-diagonal block
                                T const cij = hC(b, i, j, k);
                                auto const ii = indx2(i, k, nb, nblocks);
                                auto const jj = indx2(j, k, nb, nblocks) + nb;
                                M(ii, jj) = cij;
                            };
                        };
                    };
                }
            }; // end for k

            err = norm_error('F', n, n, n, M_.data(), MRes_.data());
            if(err > 0.1)
            {
                printf("b=%d,err from M - Mres %e, singular=%d,nb=%d,nblocks=%d,bc=%d\n", b, err,
                       singular, nb, nblocks, bc);
            };

            bool do_print = (idebug >= 1) && (err > 0.1) && (n < 32);
            if(do_print)
            {
                print_mat("hA", nb, nblocks - 1, &(hA(0, 0, 0, 0)), lda, bc);
                print_mat("hB", nb, nblocks, &(hB(0, 0, 0, 0)), ldb, bc);
                print_mat("hC", nb, nblocks - 1, &(hC(0, 0, 0, 0)), ldc, bc);
                print_mat("hBRes", nb, nblocks, &(hBRes(0, 0, 0, 0)), ldb, bc);
                print_mat("hCRes", nb, nblocks - 1, &(hCRes(0, 0, 0, 0)), ldc, bc);

                print_fullmat("M", n, n, M_.data(), ldM);
                print_fullmat("L", n, n, L_.data(), ldL);
                print_fullmat("U", n, n, U_.data(), ldU);
                print_fullmat("MRes", n, n, MRes_.data(), ldMRes);
            };

            *max_err = err > *max_err ? err : *max_err;
        }
    }; // end for bc
}

#undef Lk
#undef Uk
#undef Dk

#undef L
#undef U
#undef M
#undef MRes

#undef hA
#undef hB
#undef hC
#undef hCRes
#undef hBRes

template <typename T, typename Td, typename Ud, typename Th>
void geblttrf_npvt_interleaved_getPerfData(const rocblas_handle handle,
                                           const rocblas_int nb,
                                           const rocblas_int nblocks,
                                           Td& dA,
                                           const rocblas_int lda,
                                           Td& dB,
                                           const rocblas_int ldb,
                                           Td& dC,
                                           const rocblas_int ldc,
                                           Ud& dInfo,
                                           const rocblas_int bc,
                                           Th& hA,
                                           Th& hB,
                                           Th& hC,
                                           double* gpu_time_used,
                                           double* cpu_time_used,
                                           const rocblas_int hot_calls,
                                           const int profile,
                                           const bool profile_kernels,
                                           const bool perf,
                                           const bool singular)
{
    if(!perf)
    {
        // geblttrf_npvt_interleaved_initData<true, false, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, bc, hA,
        //                                     hB, hC, singular);

        // // cpu-lapack performance (only if not in perf mode)
        // *cpu_time_used = get_time_us_no_sync();
        // for(rocblas_int b = 0; b < bc; ++b)
        // {
        //     cpu_getrf(m, n, hA[b], lda, hIpiv[b], hInfo[b]);
        // }
        // *cpu_time_used = get_time_us_no_sync() - *cpu_time_used;
        *cpu_time_used = nan("");
    }

    geblttrf_npvt_interleaved_initData<true, false, T>(handle, nb, nblocks, dA, lda, dB, ldb, dC,
                                                       ldc, bc, hA, hB, hC, singular);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        geblttrf_npvt_interleaved_initData<false, true, T>(handle, nb, nblocks, dA, lda, dB, ldb,
                                                           dC, ldc, bc, hA, hB, hC, singular);

        CHECK_ROCBLAS_ERROR(rocsolver_geblttrf_npvt_interleaved(
            handle, nb, nblocks, dA.data(), lda, dB.data(), ldb, dC.data(), ldc, dInfo.data(), bc));
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
        geblttrf_npvt_interleaved_initData<false, true, T>(handle, nb, nblocks, dA, lda, dB, ldb,
                                                           dC, ldc, bc, hA, hB, hC, singular);

        start = get_time_us_sync(stream);
        rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA.data(), lda, dB.data(), ldb,
                                            dC.data(), ldc, dInfo.data(), bc);
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_geblttrf_npvt_interleaved(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int nb = argus.get<rocblas_int>("nb");
    rocblas_int nblocks = argus.get<rocblas_int>("nblocks");
    rocblas_int lda = argus.get<rocblas_int>("lda", nb);
    rocblas_int ldb = argus.get<rocblas_int>("ldb", nb);
    rocblas_int ldc = argus.get<rocblas_int>("ldc", nb);

    rocblas_int bc = argus.batch_count;
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // determine sizes
    size_t const size_A = max(1, bc * size_t(lda) * nb * (nblocks - 1));
    size_t const size_B = max(1, bc * size_t(ldb) * nb * nblocks);
    size_t const size_C = max(1, bc * size_t(ldc) * nb * (nblocks - 1));
    size_t const n4_A = 1;
    size_t const n4_B = 1;
    size_t const n4_C = 1;
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_BRes = (argus.unit_check || argus.norm_check) ? size_B : 0;
    size_t size_CRes = (argus.unit_check || argus.norm_check) ? size_C : 0;

    // check invalid sizes
    bool invalid_size = (nb < 0 || nblocks < 0 || lda < nb || ldb < nb || ldc < nb || bc < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, (T*)nullptr,
                                                                  lda, (T*)nullptr, ldb, (T*)nullptr,
                                                                  ldc, (rocblas_int*)nullptr, bc),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, (T*)nullptr, lda,
                                                              (T*)nullptr, ldb, (T*)nullptr, ldc,
                                                              (rocblas_int*)nullptr, bc));

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
    host_strided_batch_vector<T> hA(size_A, 1, size_A, n4_A);
    host_strided_batch_vector<T> hB(size_B, 1, size_B, n4_B);
    host_strided_batch_vector<T> hC(size_C, 1, size_C, n4_C);
    host_strided_batch_vector<T> hBRes(size_BRes, 1, size_BRes, n4_B);
    host_strided_batch_vector<T> hCRes(size_CRes, 1, size_CRes, n4_C);
    host_strided_batch_vector<rocblas_int> hInfo(1, 1, 1, bc);
    host_strided_batch_vector<rocblas_int> hInfoRes(1, 1, 1, bc);
    device_strided_batch_vector<T> dA(size_A, 1, size_A, n4_A);
    device_strided_batch_vector<T> dB(size_B, 1, size_B, n4_B);
    device_strided_batch_vector<T> dC(size_C, 1, size_C, n4_C);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_B)
        CHECK_HIP_ERROR(dB.memcheck());
    if(size_C)
        CHECK_HIP_ERROR(dC.memcheck());

    device_strided_batch_vector<rocblas_int> dInfo(1, 1, 1, bc);
    CHECK_HIP_ERROR(dInfo.memcheck());

    // check quick return
    if(nb == 0 || nblocks == 0 || bc == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_geblttrf_npvt_interleaved(handle, nb, nblocks, dA.data(),
                                                                  lda, dB.data(), ldb, dC.data(),
                                                                  ldc, dInfo.data(), bc),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        geblttrf_npvt_interleaved_getError<T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc, dInfo,
                                              bc, hA, hB, hBRes, hC, hCRes, hInfo, hInfoRes,
                                              &max_error, argus.singular);

    // collect performance data
    if(argus.timing)
        geblttrf_npvt_interleaved_getPerfData<T>(handle, nb, nblocks, dA, lda, dB, ldb, dC, ldc,
                                                 dInfo, bc, hA, hB, hC, &gpu_time_used,
                                                 &cpu_time_used, hot_calls, argus.profile,
                                                 argus.profile_kernels, argus.perf, argus.singular);

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
            rocsolver_bench_output("nb", "nblocks", "lda", "ldb", "ldc", "batch_c");
            rocsolver_bench_output(nb, nblocks, lda, ldb, ldc, bc);
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

#define EXTERN_TESTING_GEBLTTRF_NPVT_INTERLEAVED(...) \
    extern template void testing_geblttrf_npvt_interleaved<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GEBLTTRF_NPVT_INTERLEAVED, FOREACH_SCALAR_TYPE, APPLY_STAMP)
