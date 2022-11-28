/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "rocsolver_getrf_npvt_strided_batched.hpp"
#include "rocsolver_getrs_npvt_strided_batched.hpp"
#include "rocsolver_gemm_strided_batched.hpp"


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

template<typename T, typename I, typename Istride >
rocblas_status rocsolver_geblttrf_npvt_strided_batched_large_template(
	rocblas_handle handle,
	const I nb,
	const I nblocks,
        T* A_, 
        const I lda,
        const Istride strideA,
        T* B_,
        const I ldb,
        const Istride strideB,
        T* C_,
        const I ldc,
        const Istride strideC,
        I info_array[], // array of batch_count integers on GPU
        const I batch_count )
{

/*
 -----------------
 arrays dimensioned as 

 A(lda,nb,nblocks, batch_count)
 B(ldb,nb,nblocks, batch_count)
 C(ldc,nb,nblocks, batch_count)
 -----------------
*/



#ifndef indx3
// i1 + i2*n1 + i3*(n1*n2)
#define indx3(i1,i2,i3, n1,n2) \
  (((i3)*( (int64_t) (n2)) + (i2))*(n1) + (i1))
#endif

#ifndef indx3f
#define indx3f(i1,i2,i3, n1,n2) \
        indx3( ((i1)-1), ((i2)-1), ((i3)-1), n1,n2)
#endif
	

#define A(i1,i2,i3,ibatch)  \
        A_[ ((ibatch)-1)*strideA + indx3f(i1,i2,i3, lda,nb) ]

#define B(i1,i2,i3,ibatch ) \
	B_[ ((ibatch)-1)*strideB + indx3f(i1,i2,i3, ldb,nb) ]

#define C(i1,i2,i3,ibatch)  \
	C_[ ((ibatch)-1)*strideC + indx3f(i1,i2,i3, ldc,nb) ]



/*
!     --------------------------
!     reuse storage
!     over-write matrix B with D
!     over-write matrix C with U
!     --------------------------
*/
#define D(i1,i2,i3,i4) B(i1,i2,i3,i4)
#define U(i1,i2,i3,i4) C(i1,i2,i3,i4)
I const ldd = ldb;
Istride const strideD = strideB;

I const ldu = ldc;
Istride const strideU = strideC;

 
 





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
 Istride const stride1 = strideD;

 I const i = 1;
 I const j = 1;
 I const ibatch = 1;
 T* Ap = &(D(i,j,k,ibatch));
 rocblas_status istat = rocsolver_getrf_npvt_strided_batched( 
       handle, mm,nn,Ap,ld1,stride1, info_array, batch_count );
 if (istat != rocblas_status_success) {
   return( istat );
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
  for(I k=1; k <= (nblocks-1); k++) {

     {
/*
     -------------------------------------------------------------
     U(1:nb,1:nb,k) = getrs_npvt( D(1:nb,1:nb,k), C(1:nb,1:nb,k) );

     Note reuse storage for C(:,:,:,:) as U(:,:,:,:)
     -------------------------------------------------------------
*/

 
      I const nn = nb;
      I const nrhs = nb;
        
      I const ibatch = 1;
      T const * const Ap = &(D(1,1,k,ibatch));
      I const ld1 = ldd;
      Istride const stride1 = strideD;

      T * const Bp = &(U(1,1,k,ibatch));
      I const ld2 = ldu;
      Istride const stride2 = strideU;

     
      rocblas_status istat = rocsolver_getrs_npvt_strided_batched( 
                  handle, nn, nrhs,
                  Ap, ld1, stride1,
                  Bp, ld2, stride2,
                  batch_count );
      if (istat != rocblas_status_success) {
        return( istat );
        };
      };

      
/*
!--------------------------------------------
!    D(1:nb,1:nb,k+1) = B(1:nb,1:nb,k+1) - 
!          A(1:nb,1:nb,k+1) * U(1:nb,1:nb,k);
!--------------------------------------------
*/
    {
    T const alpha = -1;
    T const beta = 1;
    I const ibatch = 1;

    I const mm = nb;
    I const nn = nb;
    I const kk = nb;

    T const * const Ap = &(A(1,1,k+1,ibatch));
    I const ld1 = lda;
    Istride const stride1 = strideA;

    T const * const Bp = &(U(1,1,k,ibatch));
    I const ld2 = ldu;
    I const stride2 = strideU;

    T* const Cp = &(B(1,1,k+1,ibatch));
    I const ld3 = ldb;
    Istride const stride3 = strideB;


    rocblas_operation const transA = rocblas_operation_none;
    rocblas_operation const transB = rocblas_operation_none;

    rocblas_status istat = rocsolver_gemm_strided_batched(
          handle, 
                  transA, transB,
                  mm,nn,kk, 
          &alpha,  Ap, ld1, stride1,
                  Bp, ld2, stride2,
          &beta,   Cp, ld3, stride3,
          batch_count
          );
    if (istat != rocblas_status_success) {
       return( istat );
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

     I const ibatch = 1;
     T* const Ap = &(D(1,1,k+1,ibatch));
     I const ld1 = ldd;
     Istride const stride1 = strideD;

     rocblas_status istat = rocsolver_getrf_npvt_strided_batched( 
            handle, mm,nn,
            Ap,ld1,stride1, 
            info_array, batch_count );
     if (istat != rocblas_status_success) {
        return( istat );
        };

     };

  }; // end for k

 return( rocblas_status_success );
};







#undef indx3
#undef indx3f
#undef A
#undef B
#undef C
#undef D
#undef U
