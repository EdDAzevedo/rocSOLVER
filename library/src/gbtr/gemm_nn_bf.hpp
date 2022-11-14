/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#ifndef GEMM_NN_BF_HPP
#define GEMM_NN_BF_HPP

#include "gbtr_common.h"

template <typename T>
DEVICE_FUNCTION void gemm_nn_bf_device(rocblas_int const batchCount,
                                       rocblas_int const m,
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
#define A(iv, ia, ja) A_[indx3f(iv, ia, ja, batchCount, lda)]
#define B(iv, ib, jb) B_[indx3f(iv, ib, jb, batchCount, ldb)]
#define C(iv, ic, jc) C_[indx3f(iv, ic, jc, batchCount, ldc)]

#ifdef USE_GPU
    rocblas_int const iv_start = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    rocblas_int const iv_end = batchCount;
    rocblas_int const iv_inc = (gridDim.x * blockDim.x);
#else
    rocblas_int const iv_start = 1;
    rocblas_int const iv_end = batchCount;
    rocblas_int const iv_inc = 1;
#endif

    T const zero = 0;

    bool const is_beta_zero = (beta == zero);

    for(rocblas_int jc = 1; jc <= n; jc++)
    {
        for(rocblas_int ic = 1; ic <= n; ic++)
        {
            for(rocblas_int iv = iv_start; iv <= iv_end; iv += iv_inc)
            {
                T cij = zero;
                for(rocblas_int ja = 1; ja <= k; ja++)
                {
                    cij += A(iv, ic, ja) * B(iv, ja, jc);
                };

                if(is_beta_zero)
                {
                    C(iv, ic, jc) = alpha * cij;
                }
                else
                {
                    C(iv, ic, jc) = beta * C(iv, ic, jc) + alpha * cij;
                };
            }; // end for iv

            SYNCTHREADS;

        }; // end for ic
    }; // end for jc
}

#undef A
#undef B
#undef C

#endif
