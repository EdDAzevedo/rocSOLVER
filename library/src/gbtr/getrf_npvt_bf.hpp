/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/
#ifndef GETRF_NPVT_BF_HPP
#define GETRF_NPVT_BF_HPP

#include "geblt_common.h"

template <typename T, typename I>
DEVICE_FUNCTION void
    getrf_npvt_bf_device(I const batchCount, I const m, I const n, T* A_, I const lda, I info[])
{
    I const min_mn = (m < n) ? m : n;
    T const one = 1;

#ifdef USE_GPU
    I const iv_start = (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    I const iv_end = batchCount;
    I const iv_inc = (gridDim.x * blockDim.x);
#else
    I const iv_start = 1;
    I const iv_end = batchCount;
    I const iv_inc = 1;
#endif

#define A(iv, i, j) A_[indx3f(iv, i, j, batchCount, lda)]

    T const zero = 0;

    for(I j = 1; j <= min_mn; j++)
    {
        I const jp1 = j + 1;

        for(I iv = iv_start; iv <= iv_end; iv += iv_inc)
        {
            bool const is_diag_zero = (std::abs(A(iv, j, j)) == zero);
            T const Ujj_iv = is_diag_zero ? one : A(iv, j, j);
            info[iv - 1] = is_diag_zero && (info[iv - 1] == 0) ? j : info[iv - 1];

            for(I ia = jp1; ia <= m; ia++)
            {
                A(iv, ia, j) = A(iv, ia, j) / Ujj_iv;
            };
        };

        SYNCTHREADS;

        for(I ja = jp1; ja <= n; ja++)
        {
            for(I ia = jp1; ia <= m; ia++)
            {
                for(I iv = iv_start; iv <= iv_end; iv += iv_inc)
                {
                    A(iv, ia, ja) = A(iv, ia, ja) - A(iv, ia, j) * A(iv, j, ja);
                };
            };
        };

        SYNCTHREADS;
    };
}
#undef A

#endif
