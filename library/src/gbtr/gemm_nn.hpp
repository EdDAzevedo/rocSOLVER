/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#ifndef GEMM_NN_HPP
#define GEMM_NN_HPP

#include "gbtr_common.h"
#include "gemm_nn_fixed.hpp"

template <typename T>
DEVICE_FUNCTION void gemm_nn_device(rocblas_int const m,
                                    rocblas_int const n,
                                    rocblas_int const k,
                                    T const alpha,
                                    T const* const A_,
                                    rocblas_int const lda,
                                    T const* const B_,
                                    rocblas_int const ldb,
                                    T const beta,
                                    T* C_,
                                    rocblas_int const ldc)
{
#define A(ia, ja) A_[indx2f(ia, ja, lda)]
#define B(ib, jb) B_[indx2f(ib, jb, ldb)]
#define C(ic, jc) C_[indx2f(ic, jc, ldc)]

#define FIXED_CASE(M, N)                                                              \
    {                                                                                 \
        if((m == (M)) && (n == (N)))                                                  \
        {                                                                             \
            gemm_nn_fixed_device<T, M, N>(k, alpha, A_, lda, B_, ldb, beta, C_, ldc); \
        };                                                                            \
        return;                                                                       \
    }

    // check for special cases

    FIXED_CASE(1, 1);
    FIXED_CASE(2, 2);
    FIXED_CASE(3, 3);
    FIXED_CASE(4, 4);
    FIXED_CASE(5, 5);
    FIXED_CASE(6, 6);
    FIXED_CASE(7, 7);
    FIXED_CASE(8, 8);
    FIXED_CASE(9, 9);
    FIXED_CASE(10, 10);

    // code for general case

    T const zero = 0;
    bool const is_beta_zero = (beta == zero);

    for(rocblas_int jc = 1; jc <= n; jc++)
    {
        for(rocblas_int ic = 1; ic <= m; ic++)
        {
            T cij = zero;
            for(rocblas_int ja = 1; ja <= k; ja++)
            {
                cij += A(ic, ja) * B(ja, jc);
            };
            if(is_beta_zero)
            {
                C(ic, jc) = alpha * cij;
            }
            else
            {
                C(ic, jc) = beta * C(ic, jc) + alpha * cij;
            };
        };
    };
}

#undef A
#undef B
#undef C

#undef FIXED_CASE
#endif
