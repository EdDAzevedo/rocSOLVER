/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#ifndef GEMM_NN_FIXED_HPP
#define GEMM_NN_FIXED_HPP

#include "gbtr_common.h"

template <typename T, int const M, int const N>
DEVICE_FUNCTION void gemm_nn_fixed_device(rocblas_int const k,
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
    T const zero = 0;
    bool const is_beta_zero = (beta == zero);

#pragma unroll
    for(rocblas_int jc = 1; jc <= N; jc++)
    {
#pragma unroll
        for(rocblas_int ic = 1; ic <= M; ic++)
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

#endif
