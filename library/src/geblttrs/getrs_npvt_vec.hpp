/*
! -------------------------------------------------------------------
! Copyright(c) 2019-2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/
auto getrs_npvt_vec = [=]( rocblas_int const n,
		           rocblas_int const nrhs,
			   T *A_, rocblas_int const lda,
			   T *B_, rocblas_int const ldb) -> rocblas_int {

/*
!     ---------------------------------------------------
!     Perform forward and backward solve without pivoting
!     ---------------------------------------------------
*/
	rocblas_int const ncolA = n;
	rocblas_int const ncolB = nrhs;

#include "indx3f.hpp"
#include "A3array.hpp"
#include "B3array.hpp"
#include "syncthreads.hpp"

      T const one = 1;
      rocblas_int info = 0;

#ifdef USE_CPU
      rocblas_int const iv_start = 1;
      rocblas_int const iv_inc = 1;
#else
      rocblas_int const iv_start = hipThreadIdx_x;
      rocblas_int const iv_inc = hipBlockDim_x;
#endif

/*
! 
! % ------------------------
! % L * (U * X) = B
! % step 1: solve L * Y = B
! % step 2: solve U * X = Y
! % ------------------------
! 
*/

/*
! 
! % ------------------------------
! % [I         ] [ Y1 ]   [ B1 ]
! % [L21 I     ] [ Y2 ] = [ B2 ]
! % [L31 L21 I ] [ Y3 ]   [ B3 ]
! % ------------------------------
*/


/*
! ----------------------------------------
! for i=1:n,
!   for j=1:(i-1),
!     B(1:nvec,i,1:nrhs) = B(1:nvec,i,1:nrhs) - 
!                             LU(1:nvec,i,j) * B(1:nvec,j,1:nrhs);
!   end;
! end;
! ----------------------------------------
*/

      for(rocblas_int i=1; i <= n; i++) {

#ifndef USE_CPU
         --syncthreads();
#endif

        for(rocblas_int j=1; j <= (i-1); j++) {
	 for(rocblas_int k=1; k <= nrhs; k++) {
           for(rocblas_int iv=iv_start; iv <= nvec; iv += iv_inc) {
		 B(iv,i,k) = B(iv,i,k) - A(iv,i,j) * B(iv,j,k);
	         };
	   };

         }; // for j
      }; // for i


#ifndef USE_CPU
         --syncthreads();
#endif

/*
! 
! % ------------------------------
! % [U11 U12 U13 ] [ X1 ] = [ Y1 ]
! % [    U22 U23 ]*[ X2 ] = [ Y2 ]
! % [        U33 ]*[ X3 ] = [ Y3 ]
! % ------------------------------
! 
*/

/*
! ------------------------------------
! for ir=1:n,
!   i = n - ir + 1;
!   for j=(i+1):n,
!     B(i,1:nrhs) = B(i,1:nrhs) - LU( i,j) * B(j,1:nrhs);
!   end;
!   B(i,1:nrhs) = B(i,1:nrhs) / LU(i,i);
! end;
! 
! ------------------------------------
*/


      for(rocblas_int ir=1; ir <= n; ir++) {
	      rocblas_int const i = n - ir + 1;

	      SYNCTHREADS();

	      for(rocblas_int j=(i+1); j <= n; j++) {
		 for(rocblas_int k=1; k <= nrhs; k++) {
	           for(rocblas_int iv=iv_start; iv <= nvec; iv += iv_inc) {
			B(iv,i,k) = B(iv,i,k) - A(iv,i,j)*B(iv,j,k);
		   };
		 };
	      };

	      SYNCTHREADS();


	      for(rocblas_int iv=iv_start; iv <= nvec; iv += iv_inc) {
		      T const A_iv_i_i = A(iv,i,i);
		      bool const is_zero = (A_iv_i_i == 0);

		      T const Uii_iv = (is_zero) ? one : A_iv_i_i;
		      info = (is_zero && (info == 0)) ? i : info;

		      T const inv_Uii_iv = one/Uii_iv;

		      for(rocblas_int k=1; k <= nrhs; k++) {
			      B(iv,i,k) *= inv_Uii_iv;
		      };
	      };
      }; // for ir




      return(info);
};
