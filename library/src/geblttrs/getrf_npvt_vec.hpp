/*
! -------------------------------------------------------------------
! Copyright(c) 2019-2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/


auto getrf_npvt_vec = [=]( rocblas_int const m,
			   rocblas_int const n,
			   T *A_, rocblas_int const lda ) -> rocblas_int {
/*
!     ----------------------------------------
!     Perform LU factorization without pivoting
!     Matrices L and U over-writes matrix A
!     ----------------------------------------
*/
	auto min  = [=](rocblas_int const m, 
			rocblas_int const n ) -> rocblas_int {
		return(  (m < n) ? m : n );
	};

	rocblas_int const ncolA = n;
#include "indx3f.hpp"
#include "A3array.hpp"
#include "syncthreads.hpp"

      rocblas_int info = 0;
      rocblas_int const min_mn = min(m,n);
/*
! 
! % ----------------------------------------------------------
! % note in actual code, L and U over-writes original matrix A
! % ----------------------------------------------------------
! for j=1:min_mn,
!   jp1 = j + 1;
! 
!   U(j,j) = A(j,j);
!   L(j,j) = 1;
! 
!   L(jp1:m,j) = A(jp1:m,j) / U(j,j);
!   U(j,jp1:n) = A(j,jp1:n);
! 
!   A(jp1:m,jp1:n) = A(jp1:m, jp1:n) - L(jp1:m,j) * U(j, jp1:n);
! end;
*/
#ifdef USE_CPU
      rocblas_int const iv_start = 1;
      rocblas_int const iv_inc = 1;
#else
      rocblas_int const iv_start = hipThreadId_x;
      rocblas_int const iv_inc = hipBlockDim_x;
#endif

      T const one = 1;

      for(rocblas_int j=1; j <= min_mn; j++) {
	      rocblas_int const jp1 = j + 1;

	      SYNCTHREADS();

/*
!        ---------------------------------
!        A(1:nvec,jp1:m,j) = A(1:nvec,jp1:m,j) * /Ujj(1:nvec)
!        ---------------------------------
*/
	      for(rocblas_int iv=iv_start; iv <= nvec; iv += iv_inc) {
		      T const A_iv_j_j = A(iv,j,j);
		      bool const is_zero = (A_iv_j_j == 0);

		      T const Ujj_iv = (is_zero)? one : A_iv_j_j;
		      info = (is_zero && (info == 0)) ? j : info;

		      T const inv_Ujj_iv = one/Ujj_iv;
		      for(rocblas_int ia=jp1; ia <= m; ia++) {
			      A(iv,ia,j) = A(iv,ia,j) * inv_Ujj_iv;
		      };
	      };


	      for(rocblas_int ja=jp1; ja <= n; ja++) {
              for(rocblas_int ia=jp1; ia <= m; ia++) {
                for(rocblas_int iv=iv_start; iv <= nvec; iv += iv_inc) {
			A(iv,ia,ja) = A(iv,ia,ja) - A(iv,ia,j)*A(iv,j,ja);
		};
	      };
	      };
              

      }; // for j


#ifndef USE_CPU
      __syncthreads();
#endif
      return(info);
};
