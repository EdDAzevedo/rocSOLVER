/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#ifndef GETRS_NPVT_HPP
#define GETRS_NPVT_HPP

#include "gbtr_common.h"

template< typename T>
DEVICE_FUNCTION void
getrs_npvt_device
(
rocblas_int const n,
rocblas_int const nrhs,
T const * const A_,
rocblas_int const lda,
T *B_,
rocblas_int const ldb,
rocblas_int *pinfo
)
{
#define A(ia,ja) A_[ indx2f(ia,ja,lda) ]
#define B(ib,jb) B_[ indx2f(ib,jb,ldb) ]

T const zero = 0;
T const one = 1;
rocblas_int info = 0;
/*
!     ---------------------------------------------------
!     Perform forward and backward solve without pivoting
!     ---------------------------------------------------
*/
/*
! 
! % ------------------------
! % L * (U * X) = B
! % step 1: solve L * Y = B
! % step 2: solve U * X = Y
! % ------------------------
*/


/*
! 
! 
! % ------------------------------
! % [I         ] [ Y1 ]   [ B1 ]
! % [L21 I     ] [ Y2 ] = [ B2 ]
! % [L31 L21 I ] [ Y3 ]   [ B3 ]
! % ------------------------------
*/

/*
! 
! for i=1:n,
!   for j=1:(i-1),
!     B(i,1:nrhs) = B(i,1:nrhs) - LU(i,j) * B(j,1:nrhs);
!   end;
! end;
*/

  for(rocblas_int i=1; i <= n; i++) {
  for(rocblas_int j=1; j <= (i-1); j++) {
    for(rocblas_int k=1; k <= nrhs; k++) {
      B(i,k) = B(i,k) - A(i,j) * B(j,k);
      };
   };
   };


/*
! 
! % ------------------------------
! % [U11 U12 U13 ] [ X1 ] = [ Y1 ]
! % [    U22 U23 ]*[ X2 ] = [ Y2 ]
! % [        U33 ]*[ X3 ] = [ Y3 ]
! % ------------------------------
! 
! for ir=1:n,
!   i = n - ir + 1;
!   for j=(i+1):n,
!     B(i,1:nrhs) = B(i,1:nrhs) - LU( i,j) * B(j,1:nrhs);
!   end;
!   B(i,1:nrhs) = B(i,1:nrhs) / LU(i,i);
! end;
! 
*/

   for(rocblas_int ir=1; ir <= n; ir++) {
     rocblas_int const i = n - ir + 1;

     for(rocblas_int j=(i+1); j <= n; j++) {
       for(rocblas_int k=1; k <= nrhs; k++) {
          B(i,k) = B(i,k) - A(i,j) * B(j,k);
	  };
       };

      bool const is_diag_zero = (A(i,i) == zero);
      T const inv_Uii = (is_diag_zero) ? one : one/A(i,i);
      info = is_diag_zero && (info == 0) ?  i : info;

      for(rocblas_int k=1; k <= nrhs; k++) {
          B(i,k) *=  inv_Uii;
	  };

      };


  if (info != 0) {
    *pinfo = info;
    };

};

#undef A
#undef B

#endif

