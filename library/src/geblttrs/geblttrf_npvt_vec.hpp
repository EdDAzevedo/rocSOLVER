/*
! -------------------------------------------------------------------
! Copyright(c) 2019-2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/
#include <cstddef>
#include <cassert>
#ifdef USE_CPU
typedef int rocblas_int;
#else
#include "rocblas.hpp"
#endif

auto geblttrf_npvt_vec = [=]<typename T>( 
			      rocblas_int const nvec,
		              rocblas_int const nb,
			      rocblas_int const nblocks,
			      T *A_, rocblas_int const lda,
                              T *B_, rocblas_int const ldb,
			      T *C_, rocblas_int const ldc ) -> rocblas_int {

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

	rocblas_int info = 0;

#include "indx4f.hpp"
#include "A4array.hpp"
#include "B4array.hpp"
#include "C4array.hpp"
#include "syncthreads.hpp"

#include "getrf_npvt_vec.hpp"
#include "getrs_npvt_vec.hpp"
#include "gemm_nn_vec.hpp"





/*
!     --------------------------
!     reuse storage
!     over-write matrix B with D
!     over-write matrix C with U
!     --------------------------
*/
	rocblas_int const ldd = ldb;
	rocblas_int const ldu = ldc;

	auto D = [=](   rocblas_int const iv, 
			rocblas_int const i,
			rocblas_int const j,
			rocblas_int const k) -> T& {
		return( B(iv,i,j,k) );
	};
	auto U = [=]( rocblas_int const iv,
		      rocblas_int const i,
		      rocblas_int const j,
		      rocblas_int const k) -> T& {
		return( C(iv,i,j,k) );
	};


/*
! 
! % B1 = D1
! % D1 * U1 = C1 => U1 = D1 \ C1
! % D2 + A2*U1 = B2 => D2 = B2 - A2*U1
! %
! % D2*U2 = C2 => U2 = D2 \ C2
! % D3 + A3*U2 = B3 => D3 = B3 - A3*U2
! %
! % D3*U3 = C3 => U3 = D3 \ C3
! % D4 + A4*U3 = B4 => D4 = B4 - A4*U3
! 
*/


/* 
!--------------------------------
! 
! k = 1;
! D(:,1:nb,1:nb,k) = B(:,1:nb,1:nb,k);
! if (use_getrf_npvt),
!   D(:,1:nb,1:nb,k) = getrf_npvt( D(:,1:nb,1:nb,k) );
! end;
!--------------------------------
*/

      {
      rocblas_int const iv = 1;
      rocblas_int const k = 1;
      rocblas_int const mm = nb;
      rocblas_int const nn = nb;
/*
!   ----------------------------------------------
!   D(:,1:nb,1:nb,k) = getrf_npvt( D(:,1:nb,1:nb,k) );
!   ----------------------------------------------
*/
      rocblas_int const linfo = getrf_npvt_vec( nvec, mm,nn,&(D(iv,1,1,k)), ldd );
      info = ((linfo != 0) && (info == 0)) ? (k-1)*nb+linfo : info;
      };


/*
!--------------------------------   
! 
! for k=1:(nblocks-1),
!    
!    if (use_getrf_npvt),
!     if (idebug >= 2),
!       Ck = C(1:nb,1:nb,k);
!       disp(sprintf('k=%d,size(Ck)=%d,%d ',k,size(Ck,1),size(Ck,2)));
!     end;
! 
!     U(:,1:nb,1:nb,k) = getrs_npvt( D(:,1:nb,1:nb,k), C(1:nb,1:nb,k) );
!    else
!     U(:,1:nb,1:nb,k) = D(:,1:nb,1:nb,k) \ C(1:nb,1:nb,k);
!    end;
! 
!    D(:,1:nb,1:nb,k+1) = B(:,1:nb,1:nb,k+1) - A(1:nb,1:nb,k+1) * U(:,1:nb,1:nb,k);
!    if (use_getrf_npvt),
!      D(:,1:nb,1:nb,k+1) = getrf_npvt( D(:,1:nb,1:nb,k+1) );
!    end;
! end;
!--------------------------------   
*/

       for( rocblas_int k=1; k <= (nblocks-1); k++) {

	       SYNCTHREADS();
/*
!     --------------------------------------------------------------     
!     U(:,1:nb,1:nb,k) = getrs_npvt( D(:,1:nb,1:nb,k), C(1:nb,1:nb,k) );
!     --------------------------------------------------------------     
*/
         {
         rocblas_int const nn = nb;
         rocblas_int const nrhs = nb;
         rocblas_int const iv = 1;
	 rocblas_int const linfo = getrs_npvt_vec( 
			              nvec, nn,nrhs,
				      &(D(iv,1,1,k)), ldd,
				      &(C(iv,1,1,k)), ldc );
	 info = ((linfo != 0) && (info == 0)) ? (k-1)*nb+linfo : info;
	 };

/*
!    ------------------------------------------------------------------------
!    D(:,1:nb,1:nb,k+1) = B(1:nb,1:nb,k+1) - A(1:nb,1:nb,k+1) * U(:,1:nb,1:nb,k);
!    ------------------------------------------------------------------------
*/

	 {
          rocblas_int const iv = 1;
          rocblas_int const mm = nb;
          rocblas_int const nn = nb;
          rocblas_int const kk = nb;
          T const alpha = -1;
          T const beta =  1;
          rocblas_int const ld1 = lda;
          rocblas_int const ld2 = ldu;
          rocblas_int const ld3 = ldd;

	  rocblas_int const linfo = gemm_nn_vec( nvec, mm,nn,kk,
			  alpha, &(A(iv,1,1,k+1)), ld1,
			         &(U(iv,1,1,k)),   ld2,
	                  beta,  &(D(iv,1,1,k+1)), ld3 );

	  info = ((linfo != 0) && (info == 0)) ? linfo : info;
	 };


/*
!      --------------------------------------------------
!      D(:,1:nb,1:nb,k+1) = getrf_npvt( D(:,1:nb,1:nb,k+1) );
!      --------------------------------------------------
*/


	  {
          rocblas_int const iv = 1;
          rocblas_int const mm = nb;
          rocblas_int const nn = nb;

	  rocblas_int const linfo = getrf_npvt_vec(
			               nvec,mm,nn,
				       &(D(iv,1,1,k+1)), ldd );
	  info = ((linfo != 0) && (info == 0)) ? linfo : info;

	  };

       }; // for k

       SYNCTHREADS();

       return(info);
};

