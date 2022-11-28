/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocsolver_gemm_strided_batched.hpp"
#include "rocsolver_getrf_npvt_strided_batched.hpp"
#include "rocsolver_getrs_npvt_strided_batched.hpp"

/*
! ------------------------------------------------------
!     Perform LU factorization without pivoting
!     of block tridiagonal matrix
! % [B1, C1, 0      ]   [ D1         ]   [ I  U1       ]
! % [A2, B2, C2     ] = [ A2 D2      ] * [    I  U2    ]
! % [    A3, B3, C3 ]   [    A3 D3   ]   [       I  U3 ]
! % [        A4, B4 ]   [       A4 D4]   [          I4 ]
! ------------------------------------------------------
*/

template <typename T, typename I, typename Istride>
rocblas_status rocsolver_geblttrs_npvt_strided_batched_large_template(rocblas_handle handle,
                                                                      I nb,
                                                                      I nblocks,
                                                                      I nrhs,

                                                                      T* A_,
                                                                      I lda,
                                                                      Istride strideA,
                                                                      T* B_,
                                                                      I ldb,
                                                                      Istride strideB,
                                                                      T* C_,
                                                                      I ldc,
                                                                      Istride strideC,

                                                                      T* X_,
                                                                      I ldx,
                                                                      I strideX,
                                                                      I batch_count

)
{
    /*
 -----------------
 arrays dimensioned as 

 A(lda,nb,nblocks, batch_count)
 B(ldb,nb,nblocks, batch_count)
 C(ldc,nb,nblocks, batch_count)

 X(ldx,nblocks, nrhs, batch_count)
 Y(ldy,nblocks, nrhs, batch_count)

 or as
 X(ldx2, nrhs, batch_count), where ldx2 = (ldx * nblocks)
 Y(ldy2, nrhs, batch_count), where ldy2 = (ldy * nblocks)
 -----------------
*/

#ifndef indx3
// i1 + i2*n1 + i3*(n1*n2)
#define indx3(i1, i2, i3, n1, n2) (((i3) * ((int64_t)(n2)) + (i2)) * (n1) + (i1))
#endif

#ifndef indx3f
#define indx3f(i1, i2, i3, n1, n2) indx3(((i1)-1), ((i2)-1), ((i3)-1), n1, n2)
#endif

#define A(i1, i2, i3, ibatch) A_[((ibatch)-1) * strideA + indx3f(i1, i2, i3, lda, nb)]

#define B(i1, i2, i3, ibatch) B_[((ibatch)-1) * strideB + indx3f(i1, i2, i3, ldb, nb)]

#define C(i1, i2, i3, ibatch) C_[((ibatch)-1) * strideC + indx3f(i1, i2, i3, ldc, nb)]

#define X(i1, i2, i3, ibatch) X_[((ibatch)-1) * strideX + indx3f(i1, i2, i3, ldx, nb)]

/*
!     --------------------------
!     reuse storage
!     over-write matrix B with D
!     over-write matrix C with U
!     --------------------------
*/
#define D(i1, i2, i3, i4) B(i1, i2, i3, i4)
#define U(i1, i2, i3, i4) C(i1, i2, i3, i4)
    rocblas_int const ldd = ldb;
    rocblas_stride const strideD = strideB;

    rocblas_int const ldu = ldc;
    rocblas_stride const strideU = strideC;

#define Y(i1, i2, i3, i4) X(i1, i2, i3, i4)
    rocblas_stride const strideY = strideX;
    rocblas_int const ldy = ldx;

    rocblas_int const ldx2 = ldx * nblocks;
    rocblas_int const ldy2 = ldy * nblocks;

    /*
! 
! % forward solve
! % --------------------------------
! % [ D1          ]   [ y1 ]   [ b1 ]
! % [ A2 D2       ] * [ y2 ] = [ b2 ]
! % [    A3 D3    ]   [ y3 ]   [ b3 ]
! % [       A4 D4 ]   [ y4 ]   [ b4 ]
! % --------------------------------
! %
! % ------------------
! % y1 = D1 \ b1
! % y2 = D2 \ (b2 - A2 * y1)
! % y3 = D3 \ (b3 - A3 * y2)
! % y4 = D4 \ (b4 - A4 * y3)
! % ------------------
! 
! 
*/

    /*
! for k=1:nblocks,
!     if ((k-1) >= 1),
!       y(1:nb,k,:) = y(1:nb,k,:) - A(1:nb,1:nb,k) * y(1:nb,k-1,:);
!     end;
!      LU = D(1:nb,1:nb,k);
!      y(1:nb,k,:) = getrs_npvt( LU, y(1:nb,k,:) );
! end;
*/

    T const one = 1;
    I info = 0;

    for(I k = 1; k <= nblocks; k++)
    {
        if((k - 1) >= 1)
        {
            /*
!         ----------------------------------------------------
!         y(1:nb,k,1:nrhs,1:batch_count) = y(1:nb, k, 1:nrhs, 1:batch_count) - 
                          A(1:nb,1:nb,k,1:batch_count) * y(1:nb,k-1,1:nrhs,1:batch_count);
!         ----------------------------------------------------
*/
            I const mm = nb;
            I const nn = nrhs;
            I const kk = nb;

            T const alpha = -one;
            T const beta = one;

            T const* const Ap = &(A(1, 1, k, 1));
            I const ld1 = lda;
            Istride const stride1 = strideA;

            T const* const Bp = &(Y(1, k - 1, 1, 1));
            I const ld2 = ldy2;
            Istride const stride2 = strideY;

            T* const Cp = &(Y(1, k, 1, 1));
            I const ld3 = ldy2;
            Istride const stride3 = strideY;

            rocblas_operation const transA = rocblas_operation_none;
            rocblas_operation const transB = rocblas_operation_none;

            rocblas_status istat = rocsolver_gemm_strided_batched(
                handle, transA, transB, mm, nn, kk, &alpha, Ap, ld1, stride1, Bp, ld2, stride2,
                &beta, Cp, ld3, stride3, batch_count);
            if(istat != rocblas_status_success)
            {
                return (istat);
            };
        };

        /*
!      ----------------------------------------------------
!      y(1:nb,k,1:nrhs,1:batch_count) = getrs_npvt( D(1:nb,1:nb,k,1:batch_count), 
!                                                   y(1:nb,k,1:nrhs,1:batch_count) );
!      ----------------------------------------------------
*/
        {
            I const nn = nb;
            I linfo = 0;

            T const* const Ap = &(D(1, 1, k, 1));
            I const ld1 = ldd;
            Istride const stride1 = strideD;

            T* const Bp = &(Y(1, k, 1, 1));
            I const ld2 = ldy2;
            Istride const stride2 = strideY;

            rocblas_status istat = rocsolver_getrs_npvt_strided_batched(
                handle, nn, nrhs, Ap, ld1, stride1, Bp, ld2, stride2, batch_count);
            info = (linfo != 0) && (info == 0) ? (k - 1) * nb + linfo : info;
            if(istat != rocblas_status_success)
            {
                return (istat);
            };
        };
    }; // end for k

    /*
! 
! % backward solve
! % ---------------------------------
! % [ I  U1       ]   [ x1 ]   [ y1 ]
! % [    I  U2    ] * [ x2 ] = [ y2 ]
! % [       I  U3 ]   [ x3 ]   [ y3 ]
! % [          I  ]   [ x4 ]   [ y4 ]
! % ---------------------------------
! % 
! % x4 = y4
! % x3 = y3 - U3 * y4
! % x2 = y2 - U2 * y3
! % x1 = y1 - U1 * y2
! %
*/

    /*
! 
! x = zeros(nb,nblocks);
! for kr=1:nblocks,
!   k = nblocks - kr+1;
!   if (k+1 <= nblocks),
!     y(1:nb,k,:) = y(1:nb,k,:) - U(1:nb,1:nb,k) * x(1:nb,k+1,:);
!   end;
!   x(1:nb,k,:) = y(1:nb,k,:);
! end;
! 
*/

    for(I kr = 1; kr <= nblocks; kr++)
    {
        I const k = nblocks - kr + 1;
        if((k + 1) <= nblocks)
        {
            /*
!     ----------------------------------------------------------
!     y(1:nb,k,1:nrhs,1:batch_count) = 
!                 y(1:nb,k,1:nrhs,1:batch_count) - 
!                     U(1:nb,1:nb,k,1:batch_count) * x(1:nb,k+1,1:nrhs,1:batch_count);
!     ----------------------------------------------------------
*/
            I const mm = nb;
            I const nn = nrhs;
            I const kk = nb;

            T const alpha = -one;
            T const beta = one;

            T const* const Ap = &(U(1, 1, k, 1));
            I const ld1 = ldu;
            Istride const stride1 = strideU;

            T const* const Bp = &(X(1, k + 1, 1, 1));
            I const ld2 = ldx2;
            Istride const stride2 = strideX;

            T* const Cp = &(Y(1, k, 1, 1));
            I const ld3 = ldy2;
            Istride const stride3 = strideY;

            rocblas_operation const transA = rocblas_operation_none;
            rocblas_operation const transB = rocblas_operation_none;

            rocblas_status istat = rocsolver_gemm_strided_batched(
                handle, transA, transB, mm, nn, kk, &alpha, Ap, ld1, stride1, Bp, ld2, stride2,
                &beta, Cp, ld3, stride3, batch_count);

            if(istat != rocblas_status_success)
            {
                return (istat);
            };
        };
    };

    return (rocblas_status_success);
};

#undef indx3
#undef indx3f
#undef A
#undef B
#undef C
#undef D
#undef U
#undef X
#undef Y
