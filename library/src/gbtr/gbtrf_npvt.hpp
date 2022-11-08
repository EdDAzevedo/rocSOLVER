/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#ifndef GBTRF_NPVT_HPP
#define GBTRF_NPVT_HPP

#include "gbtr_common.h"

#include "gemm_nn.hpp"
#include "getrf_npvt.hpp"
#include "getrs_npvt.hpp"

template< typename T>
DEVICE_FUNCTION void
gbtrf_npvt_device
(
rocblas_int const nb,
rocblas_int const nblocks,
T *A_,
rocblas_int const lda,
T *B_,
rocblas_int const ldb,
T *C_,
rocblas_int const ldc,
rocblas_int *pinfo
)
{

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


#define A(ia,ja,iblock) A_[ indx3f(ia,ja,iblock,   lda,nb) ]
#define B(ib,jb,iblock) B_[ indx3f(ib,jb,iblock,   ldb,nb) ]
#define C(ic,jc,iblock) C_[ indx3f(ic,jc,iblock,   ldc,nb) ]


rocblas_int info = 0;
T const one = 1;
T const zero = 0;


/*
!     --------------------------
!     reuse storage
!     over-write matrix B with D
!     over-write matrix C with U
!     --------------------------
*/
#define D(i,j,k) B(i,j,k)
#define U(i,j,k) C(i,j,k)
       rocblas_int const ldu = ldc;
       rocblas_int const ldd = ldb;
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
*/

/*
! 
! % ----------------------------------
! % in actual code, overwrite B with D
! % overwrite C with U
! % ----------------------------------
! D = zeros(nb,nb,nblocks);
! U = zeros(nb,nb,nblocks);
*/

/*
! 
! k = 1;
! D(1:nb,1:nb,k) = B(1:nb,1:nb,k);
! if (use_getrf_npvt),
!   D(1:nb,1:nb,k) = getrf_npvt( D(1:nb,1:nb,k) );
! end;
*/


/*
!   ----------------------------------------------
!   D(1:nb,1:nb,k) = getrf_npvt( D(1:nb,1:nb,k) );
!   ----------------------------------------------
*/
      {
      rocblas_int const k = 1;
      rocblas_int const mm = nb;
      rocblas_int const nn = nb;
      rocblas_int linfo = 0;

      getrf_npvt_device( mm, nn, &(D(1,1,k)), ldd, &linfo );
      info = (info == 0) && (linfo != 0) ? (k-1)*nb + linfo : info;
      };
   
/*
! 
! for k=1:(nblocks-1),
!    
!    if (use_getrf_npvt),
!     if (idebug >= 2),
!       Ck = C(1:nb,1:nb,k);
!       disp(sprintf('k=%d,size(Ck)=%d,%d ',k,size(Ck,1),size(Ck,2)));
!     end;
! 
!     U(1:nb,1:nb,k) = getrs_npvt( D(1:nb,1:nb,k), C(1:nb,1:nb,k) );
!    else
!     U(1:nb,1:nb,k) = D(1:nb,1:nb,k) \ C(1:nb,1:nb,k);
!    end;
! 
!    D(1:nb,1:nb,k+1) = B(1:nb,1:nb,k+1) - A(1:nb,1:nb,k+1) * U(1:nb,1:nb,k);
!    if (use_getrf_npvt),
!      D(1:nb,1:nb,k+1) = getrf_npvt( D(1:nb,1:nb,k+1) );
!    end;
! end;
*/

       for(rocblas_int k=1; k <= (nblocks-1); k++) {
/*
!     --------------------------------------------------------------     
!     U(1:nb,1:nb,k) = getrs_npvt( D(1:nb,1:nb,k), C(1:nb,1:nb,k) );
!     --------------------------------------------------------------     
*/
         {
         rocblas_int nn = nb;
         rocblas_int nrhs = nb;
         rocblas_int linfo = 0;

         getrs_npvt_device( nn,nrhs, 
	                    &(D(1,1,k)), ldd, 
			    &(C(1,1,k)),ldc, 
			    &linfo );
        
	 info = (info == 0) && (linfo != 0) ? (k-1)*nb + linfo : info;
         };

/*
!    ------------------------------------------------------------------------
!    D(1:nb,1:nb,k+1) = B(1:nb,1:nb,k+1) - A(1:nb,1:nb,k+1) * U(1:nb,1:nb,k);
!    ------------------------------------------------------------------------
*/
	  {
          rocblas_int const mm = nb;
          rocblas_int const nn = nb;
          rocblas_int const kk = nb;
          T const alpha = -one;
          T const beta =  one;
          rocblas_int const ld1 = lda;
          rocblas_int const ld2 = ldu;
          rocblas_int const ld3 = ldd;
          gemm_nn_device( mm,nn,kk, 
	                  alpha, &(A(1,1,k+1)),ld1, 
			         &(U(1,1,k)),  ld2,    
			  beta,  &(D(1,1,k+1)),ld3 );
          };


/*
!      --------------------------------------------------
!      D(1:nb,1:nb,k+1) = getrf_npvt( D(1:nb,1:nb,k+1) );
!      --------------------------------------------------
*/
	 {
          rocblas_int const mm = nb;
          rocblas_int const nn = nb;
          rocblas_int linfo = 0;
          getrf_npvt_device( mm,nn, &(D(1,1,k+1)),ldd, &linfo );
	  info = (linfo != 0) && (info == 0) ? (k-1)*nb + linfo : info;

	  };

        };

        if (info != 0) {
	   *pinfo = info;
	   };


};
#undef D
#undef U


#undef A
#undef B
#undef C


#endif
