/*
! -------------------------------------------------------------------
! Copyright(c) 2019-2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/


auto getrf_npvt_vec = [=]
(
rocblas_int nvec,
rocblas_int ldnvec,

rocblas_int const m,
rocblas_int const n,
T *A_, 
rocblas_int const lda 
) -> rocblas_int {
/*
!     ----------------------------------------
!     Perform LU factorization without pivoting
!     Matrices L and U over-writes matrix A
!     ----------------------------------------
*/

	rocblas_int const ncolA = n;
#include "indx3f.hpp"
#include "A3array.hpp"

      rocblas_int info = 0;
      rocblas_int const min_mn = (m < n) ? m : n;
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
      auto const iv_start = 1;
      auto const iv_inc = 1;
#else
      auto const iv_start = 1 + hipThreadId_x;
      auto const iv_inc = hipBlockDim_x;
#endif

      T const one = 1;

      for(auto j=1; j <= min_mn; j++) {
	      auto const jp1 = j + 1;

	      SYNCTHREADS;

/*
!        ---------------------------------
!        A(1:nvec,jp1:m,j) = A(1:nvec,jp1:m,j) * /Ujj(1:nvec)
!        ---------------------------------
*/
	      for(auto iv=iv_start; iv <= nvec; iv += iv_inc) {
		      T const A_iv_j_j = A(iv,j,j);
		      bool const is_zero = (A_iv_j_j == 0);

		      T const Ujj_iv = (is_zero)? one : A_iv_j_j;
		      info = (is_zero && (info == 0)) ? j : info;

		      T const inv_Ujj_iv = one/Ujj_iv;
		      for(auto ia=jp1; ia <= m; ia++) {
			      A(iv,ia,j) = A(iv,ia,j) * inv_Ujj_iv;
		      };
	      };


	      for(auto ja=jp1; ja <= n; ja++) {
              for(auto ia=jp1; ia <= m; ia++) {
                for(auto iv=iv_start; iv <= nvec; iv += iv_inc) {
			A(iv,ia,ja) = A(iv,ia,ja) - A(iv,ia,j)*A(iv,j,ja);
		};
	      };
	      };
              

      }; // for j


      SYNCTHREADS;

      return(info);
};
