/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#ifndef GBTRS_NPVT_HPP
#define GBTRS_NPVT_HPP


#include "gbtr_common.h"

#include "gemm_nn.hpp"
#include "getrs_npvt.hpp"

template< typename T>
DEVICE_FUNCTION void
gbtrs_npvt
(
rocblas_int const nb,
rocblas_int const nblocks,
rocblas_int const nrhs,
T const * const A_,
rocblas_int const lda,
T const * const D_,
rocblas_int const ldd,
T const * const U_,
rocblas_int const ldu,
T *brhs_,
rocblas_int const ldbrhs,
rocblas_int *pinfo
)
{

/*
! % ------------------------------------------------
! % Perform forward and backward solve
! %
! %
! % [B1, C1, 0      ]   [ D1         ]   [ I  U1       ]
! % [A2, B2, C2     ] = [ A2 D2      ] * [    I  U2    ]
! % [    A3, B3, C3 ]   [    A3 D3   ]   [       I  U3 ]
! % [        A4, B4 ]   [       A4 D4]   [          I4 ]
! %
! % ----------------------
! % Solve L * U * x = brhs
! % (1) Solve L * y = brhs,
! % (2) Solve U * x = y
! % ----------------------
*/

#define A(ia,ja,iblock) A_[ indx3f(ia,ja,iblock,  lda,nb) ]
#define D(id,jd,iblock) D_[ indx3f(id,jd,iblock,  ldd,nb) ]
#define U(iu,ju,iblock) U_[ indx3f(iu,ju,iblock,  ldu,nb) ]
#define brhs(i,j,k)     brhs_[ indx3f(i,j,k,   ldbrhs,nblocks) ]

#define x(i,j,k) brhs(i,j,k)
#define y(i,j,k) brhs(i,j,k)

rocblas_int info = 0;

rocblas_int const ldx = ldbrhs;
rocblas_int const ldy = ldbrhs;

T const one = 1;
T const zero = 0;
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
!     if (use_getrf_npvt),
!      LU = D(1:nb,1:nb,k);
!      y(1:nb,k,:) = getrs_npvt( LU, y(1:nb,k,:) );
!     else
!      y(1:nb,k,:) = D(1:nb,1:nb,k) \ y(1:nb,k,:);
!     end;
! end;
*/

      for(rocblas_int k=1; k <= nblocks; k++) {
	if ((k-1) >= 1) {
/*
!         ----------------------------------------------------
!         y(1:nb,k,:) = y(1:nb,k,:) - A(1:nb,1:nb,k) * y(1:nb,k-1,:);
!         ----------------------------------------------------
*/
	  {
          rocblas_int const mm = nb;
          rocblas_int const nn = nrhs;
          rocblas_int const kk = nb;

          T const alpha = -one;
          T const beta  =  one;

          rocblas_int const ld1 = lda;
          rocblas_int const ld2 = ldy * nblocks;
          rocblas_int const ld3 = ldy * nblocks;

          gemm_nn_device( mm,nn,kk, 
	                  alpha, &(A(1,1,k)),  ld1,                   
				 &(y(1,k-1,1)),ld2,
                          beta,  &(y(1,k,1)),  ld3);
          };
        };

/*
!      ----------------------------------------------------
!      y(1:nb,k,:) = getrs_npvt( D(1:nb,1:nb,k), y(1:nb,k,:) );
!      ----------------------------------------------------
*/
	{
        rocblas_int const nn = nb;
        rocblas_int const ld1 = ldd;
        rocblas_int const ld2 = ldbrhs*nblocks;
        rocblas_int linfo = 0;

        getrs_npvt_device( nn, nrhs, 
	                   &(D(1,1,k)), ld1, 
			   &(y(1,k,1)), ld2, 
			   &linfo);
	info = (linfo != 0) && (info == 0) ? (k-1)*nb + linfo : info;

        };

      };


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

      for(rocblas_int kr=1; kr <= nblocks; kr++) {
        rocblas_int const k = nblocks - kr + 1;
	if ( (k+1)  <= nblocks) {
/*
!     ----------------------------------------------------------
!     y(1:nb,k,:) = y(1:nb,k,:) - U(1:nb,1:nb,k) * x(1:nb,k+1,:);
!     ----------------------------------------------------------
*/
          rocblas_int const mm = nb;
          rocblas_int const nn = nrhs;
          rocblas_int const kk = nb;

          T const alpha = -one;
          T const beta  =  one;

          rocblas_int const ld1 = ldu;
          rocblas_int const ld2 = ldx * nblocks;
          rocblas_int const ld3 = ldy * nblocks;

          gemm_nn_device( mm,nn,kk, 
	                  alpha, &(U(1,1,k)),  ld1,                   
				 &(x(1,k+1,1)),ld2,                   
			  beta,  &(y(1,k,1)),  ld3);
         };
        };

   if (info != 0) {
     *pinfo = info;
     };

};
#undef x
#undef y

#undef A
#undef D
#undef U

#endif
