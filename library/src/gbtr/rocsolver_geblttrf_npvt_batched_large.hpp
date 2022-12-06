/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef ROCSOLVER_GEBLTTRF_BATCHED_LARGE_H
#define ROCSOLVER_GEBLTTRF_BATCHED_LARGE_H

#include "geblt_common.h"

#include "rocsolver_gemm_batched_with_offset.hpp"
#include "rocsolver_getrs_npvt_batched_with_offset.hpp"
#include "rocsolver_getrf_npvt_batched_with_offset.hpp"

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

template <typename T, typename I>
rocblas_status rocsolver_geblttrf_npvt_batched_large_template(
    rocblas_handle handle,
    const I nb,
    const I nblocks,
    T* A_array[],
    const I lda,
    T* B_array[],
    const I ldb,
    T* C_array[],
    const I ldc,
    I info_array[], // array of batch_count integers on GPU
    const I batch_count)
{
    /*
 -----------------
 arrays dimensioned as 

 (A_array[ibatch])(lda,nb,nblocks )
 (B_array[ibatch])(ldb,nb,nblocks, batch_count)
 (C_array[ibatch])(ldc,nb,nblocks, batch_count)
 -----------------
*/

#ifndef indx3
// i1 + i2*n1 + i3*(n1*n2)
#define indx3(i1, i2, i3, n1, n2) (((i3) * ((I)(n2)) + (i2)) * (n1) + (i1))
#endif

#ifndef indx3f
#define indx3f(i1, i2, i3, n1, n2) indx3(((i1)-1), ((i2)-1), ((i3)-1), (n1), (n2))
#endif


#define A(i1, i2, i3, ibatch) (A_array[(ibatch)-1])[ indx3f(i1, i2, i3, lda, nb)]
#define B(i1, i2, i3, ibatch) (B_array[(ibatch)-1])[ indx3f(i1, i2, i3, ldb, nb)]
#define C(i1, i2, i3, ibatch) (C_array[(ibatch)-1])[ indx3f(i1, i2, i3, ldc, nb)]


/*
!     --------------------------
!     reuse storage
!     over-write matrix B with D
!     over-write matrix C with U
!     --------------------------
*/
#define D(i1, i2, i3, i4) B(i1, i2, i3, i4)
#define U(i1, i2, i3, i4) C(i1, i2, i3, i4)
    auto D_array = B_array;
    I const ldd = ldb;

    auto U_array = C_array;
    I const ldu = ldc;

    /*
---------------------------------------
! % B1 = D1
! % D1 * U1 = C1 => U1 = D1 \ C1
! % D2 + A2*U1 = B2 => D2 = B2 - A2*U1
! %
! % D2*U2 = C2 => U2 = D2 \ C2
! % D3 + A3*U2 = B3 => D3 = B3 - A3*U2
! %
! % D3*U3 = C3 => U3 = D3 \ C3
! % D4 + A4*U3 = B4 => D4 = B4 - A4*U3
---------------------------------------
*/

    // -----------------------------------------------------------------------
    // D(1:nb,1:nb,k, 1:batch_count) = getrf_npvt( D(1:nb,1:nb,k, 1:batch_count)
    // -----------------------------------------------------------------------
    {
        I const k = 1;
        I const mm = nb;
        I const nn = nb;
        I const ld1 = ldd;

        I const i = 1;
        I const j = 1;
        // T* Ap = &(D(i, j, k, ibatch));
        auto Ap_array = D_array;
        I const offset1 = indx3f(i,j,k, ldd,nb);
        rocblas_status istat = rocsolver_getrf_npvt_batched_with_offset(handle, 
                                                                    mm, nn, 
                                                                    Ap_array, offset1, ld1, 
                                                                    info_array, 
                                                                    batch_count);
        if(istat != rocblas_status_success)
        {
            return (istat);
        };
    }

    /*
!------------------------------------------------
! for k=1:(nblocks-1),
!    
! 
!     U(1:nb,1:nb,k) = getrs_npvt( D(1:nb,1:nb,k), C(1:nb,1:nb,k) );
! 
!    D(1:nb,1:nb,k+1) = B(1:nb,1:nb,k+1) - 
!          A(1:nb,1:nb,k+1) * U(1:nb,1:nb,k);
!      D(1:nb,1:nb,k+1) = getrf_npvt( D(1:nb,1:nb,k+1) );
! end;
!------------------------------------------------
*/
    for(I k = 1; k <= (nblocks - 1); k++)
    {
        {
            /*
     -------------------------------------------------------------
     U(1:nb,1:nb,k) = getrs_npvt( D(1:nb,1:nb,k), C(1:nb,1:nb,k) );

     Note reuse storage for C(:,:,:,:) as U(:,:,:,:)
     -------------------------------------------------------------
*/

            I const nn = nb;
            I const nrhs = nb;

            // T const* const Ap = &(D(1, 1, k, ibatch));
            auto Ap_array = D_array;
            I const ld1 = ldd;
            I const offset1 = indx3f(1,1,k,  ldd,nb);

            // T* const Bp = &(U(1, 1, k, ibatch));
            auto Bp_array = U_array;
            I const ld2 = ldu;
            I const offset2 = indx3f(1,1,k, ldu, nb);

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

        /*
!--------------------------------------------
!    D(1:nb,1:nb,k+1) = B(1:nb,1:nb,k+1) - 
!          A(1:nb,1:nb,k+1) * U(1:nb,1:nb,k);
!--------------------------------------------
*/
        {
            T alpha = -1;
            T beta = 1;

            I const mm = nb;
            I const nn = nb;
            I const kk = nb;

            // T const* const Ap = &(A(1, 1, k + 1, ibatch));
            auto Ap_array = A_array;

            I const ld1 = lda;
            I const offset1 = indx3f(1,1,k+1, lda,nb);

            // T const* const Bp = &(U(1, 1, k, ibatch));
            auto Bp_array = U_array;
            I const ld2 = ldu;
            I const offset2 = indx3f(1,1,k, ldu,nb);

            // T* const Cp = &(B(1, 1, k + 1, ibatch));
            auto Cp_array = B_array;
            I const ld3 = ldb;
            I const offset3 = indx3f(1,1,k+1,  ldb,nb);

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
!      --------------------------------------------------
!      D(1:nb,1:nb,k+1) = getrf_npvt( D(1:nb,1:nb,k+1) );
!      --------------------------------------------------
*/
        {
            I const mm = nb;
            I const nn = nb;

            // T* const Ap = &(D(1, 1, k + 1, ibatch));
            auto Ap_array = D_array;
            I const ld1 = ldd;
            I const offset1 = indx3f(1,1,k+1,   ldd,nb);

            rocblas_status istat = rocsolver_getrf_npvt_batched_with_offset(
                handle, mm, nn, Ap_array, offset1, ld1, 
                               info_array, batch_count);
            if(istat != rocblas_status_success)
            {
                return (istat);
            };
        };

    }; // end for k

    return (rocblas_status_success);
};

#undef A
#undef B
#undef C
#undef D
#undef U

#endif
