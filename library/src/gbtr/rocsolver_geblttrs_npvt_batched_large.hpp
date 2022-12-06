/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSOLVER_GEBLTTRS_NPVT_BATCHED_LARGE_HPP
#define ROCSOLVER_GEBLTTRS_NPVT_BATCHED_LARGE_HPP

#include "geblt_common.h"

#include "rocsolver_gemm_batched_with_offset.hpp"
#include "rocsolver_getrf_npvt_batched_with_offset.hpp"
#include "rocsolver_getrs_npvt_batched_with_offset.hpp"

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

template <typename T, typename I >
rocblas_status rocsolver_geblttrs_npvt_batched_large_template(rocblas_handle handle,
                                                                      const I nb,
                                                                      const I nblocks,
                                                                      const I nrhs,

                                                                      T* A_array[],
                                                                      const I lda,
                                                                      T* B_array[],
                                                                      const I ldb,
                                                                      T* C_array[],
                                                                      const I ldc,

                                                                      T* X_array[],
                                                                      const I ldx,
                                                                      const I batch_count

)
{
    /*
 -----------------
 arrays accessed as 

 A(ia,ja,iblock,ibatch) as (A_array[ibatch-1])[ indx3f(ia,ja,iblock, lda, nb) ]

 X(ix,iblock,irhs,ibatch) as (X_array[ibatch-1])[  indx3f(ix,iblock,irhs, ldx, nblocks) ]
 -----------------
*/

#define indx3fA(ia,ja,iblock) indx3f(ia,ja,iblock,  lda,nb)
#define indx3fB(ib,jb,iblock) indx3f(ib,jb,iblock,  ldb,nb)
#define indx3fC(ic,jc,iblock) indx3f(ic,jc,iblock,  ldc,nb)

#define indx3fX(ix,iblock,irhs) indx3f(ix,iblock,irhs,  ldx,nblocks)



/*
!     --------------------------
!     reuse storage
!     over-write matrix B with D
!     over-write matrix C with U
!     --------------------------
*/
    auto D_array = B_array;
    rocblas_int const ldd = ldb;


    auto U_array = C_array;
    rocblas_int const ldu = ldc;

    auto Y_array = X_array;
    rocblas_int const ldy = ldx;

#define indx3fD(id,jd,iblock) indx3f(id,jd,iblock, ldd, nb)
#define indx3fU(iu,ju,iblock) indx3f(iu,ju,iblock, ldu, nb)
#define indx3fY(iy,iblock,irhs) indx3f(iy,iblock,irhs,  ldy,nblocks)

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
    I const irhs = 1;

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

            T alpha = -one;
            T beta = one;

            // T const* const Ap = &(A(1, 1, k, ibatch));
            auto Ap_array = A_array;
            I const offset1 = indx3fA(1,1,k);
            I const ld1 = lda;

            // T const* const Bp = &(Y(1,  k - 1,irhs, ibatch ));
            auto Bp_array = Y_array;
            I const offset2 = indx3fY(1,k-1,irhs);
            I const ld2 = ldy;

            // T* const Cp = &(Y(1, k, 1, ibatch));
            auto Cp_array = Y_array;
            I const offset3 = indx3fY(1,k,irhs);
            I const ld3 = ldy;

            rocblas_operation const transA = rocblas_operation_none;
            rocblas_operation const transB = rocblas_operation_none;

            rocblas_status istat = rocsolver_gemm_batched_with_offset(
                handle, transA, transB, mm, nn, kk, 
                &alpha, Ap_array, offset1, ld1, 
                        Bp_array, offset2, ld2, 
                &beta,  Cp_array, offset3, ld3, 
                        batch_count);
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
         

            // T const* const Ap = &(D(1, 1, k, 1));
            auto Ap_array = D_array;
            I const offset1 = indx3fD(1,1,k);
            I const ld1 = ldd;

            // T* const Bp = &(Y(1, k, 1, 1));
            auto Bp_array = Y_array;
            I const offset2 = indx3fY(1,k,irhs);
            I const ld2 = ldy;

            rocblas_status istat = rocsolver_getrs_npvt_batched_with_offset(
                handle, nn, nrhs, 
                Ap_array, offset1, ld1, 
                Bp_array, offset2, ld2, 
                batch_count);
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

            T alpha = -one;
            T beta = one;

            // T const* const Ap = &(U(1, 1, k, 1));
            auto Ap_array = U_array;
            I const offset1 = indx3fU(1,1,k);
            I const ld1 = ldu;

            // T const* const Bp = &(X(1, k + 1, 1, 1));
            auto Bp_array = X_array;
            I const offset2 = indx3fX(1,k+1,irhs);
            I const ld2 = ldx;

            // T* const Cp = &(Y(1, k, 1, 1));
            auto Cp_array = Y_array;
            I const offset3 = indx3fY(1,k,irhs);
            I const ld3 = ldy;

            rocblas_operation const transA = rocblas_operation_none;
            rocblas_operation const transB = rocblas_operation_none;

            rocblas_status istat = rocsolver_gemm_batched_with_offset(
                handle, transA, transB, 
                        mm, nn, kk, 
                &alpha, Ap_array, offset1, ld1, 
                        Bp_array, offset2, ld2, 
                &beta,  Cp_array, offset3, ld3, 
                        batch_count);

            if(istat != rocblas_status_success)
            {
                return (istat);
            };
        };
    };

    return (rocblas_status_success);
};


#undef indx3fX
#undef indx3fY

#undef indx3fA
#undef indx3fB
#undef indx3fC
#undef indx3fD
#undef indx3fU

#endif
