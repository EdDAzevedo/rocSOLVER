/*
! -------------------------------------------------------------------
! Copyright(c) 2019-2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/
auto geblttrs_npvt_vec = [=]( rocblas_int const nvec,
		              rocblas_int const nb,
	                      rocblas_int const nblocks,
			      rocblas_int const nrhs,
			      T *A_, rocblas_int const lda,
			      T *D_, rocblas_int const ldd,
			      T *U_, rocblas_int const ldu,
			      T *brhs_, rocblas_int const ldbrhs ) -> rocblas_int {
		            	      
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
 
   rocblas_int info = 0;

#include "indx4f.hpp"
#include "A4array.hpp"
#include "D4array.hpp"
#include "U4array.hpp"

  auto brhs = [=](rocblas_int const iv, 
		  rocblas_int const i,
		  rocblas_int const j,
		  rocblas_int const k) -> T& {
	  return( brhs_[ indx4f(iv,i,j,k,   nvec,nb,nblocks,nrhs) ] );
  };


   rocblas_int const ldx = ldbrhs;
   rocblas_int const ldy = ldbrhs;

   auto x = [=](rocblas_int const iv,
                rocblas_int const i,
		rocblas_int const j,
		rocblas_int const k ) -> T& {
	   return( brhs(iv,i,j,k) );
   };

   auto y = [=](rocblas_int const iv,
                rocblas_int const i,
		rocblas_int const j,
		rocblas_int const k ) -> T& {
	   return( brhs(iv,i,j,k) );
   };





/*
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
*/


/*
! for k=1:nblocks,
!     if ((k-1) >= 1),
!       y(:,1:nb,k,:) = y(:,1:nb,k,:) - A(:,1:nb,1:nb,k) * y(:,1:nb,k-1,:);
!     end;
!     if (use_getrf_npvt),
!      LU = D(1:nb,1:nb,k);
!      y(:,1:nb,k,:) = getrs_npvt( LU, y(:,1:nb,k,:) );
!     else
!      y(:,1:nb,k,:) = D(1:nb,1:nb,k) \ y(:,1:nb,k,:);
!     end;
! end;
*/

      for(rocblas_int k=1; k <= nblocks; k++) {
        if ((k-1) >= 1) {
/*
!         ----------------------------------------------------
!         y(:,1:nb,k,:) = y(:,1:nb,k,:) - A(1:nb,1:nb,k) * y(:,1:nb,k-1,:);
!         ----------------------------------------------------
*/
          rocblas_int const iv = 1;
          rocblas_int const mm = nb;
          rocblas_int const nn = nrhs;
          rocblas_int const kk = nb;
          T const alpha = -1;
          T const beta = 1;
          rocblas_int const ld1 = lda;
          rocblas_int const ld2 = ldy * nblocks;
          rocblas_int const ld3 = ldy * nblocks;

	  gemm_nn_vec( nvec, mm,nn,kk,
		       alpha, &(A(iv,1,1,k)),  ld1,
			      &(y(iv,1,k-1,1)),ld2,
                       beta,  &(y(iv,1,k,1)),  ld3 );

         };

/*
!      ----------------------------------------------------
!      y(:,1:nb,k,:) = getrs_npvt( D(:,1:nb,1:nb,k), y(:,1:nb,k,:) );
!      ----------------------------------------------------
*/
	{
        rocblas_int const iv = 1;
        rocblas_int const nn = nb;
        rocblas_int const ld1 = ldd;
        rocblas_int const ld2 = ldbrhs*nblocks;
        rocblas_int const linfo = getrs_npvt_vec( nvec, nn, nrhs, 
			            &(D(iv,1,1,k)),ld1,                   
                                    &(y(iv,1,k,1)), ld2);

	info = ( (linfo != 0) && (info == 0)) ? (k-1)*nb+linfo : info;
	};

      };


      SYNCTHREADS;
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
! 
! x = zeros(nb,nblocks);
! for kr=1:nblocks,
!   k = nblocks - kr+1;
!   if (k+1 <= nblocks),
!     y(:,1:nb,k,:) = y(:,1:nb,k,:) - U(1:nb,1:nb,k) * x(:,1:nb,k+1,:);
!   end;
!   x(:,1:nb,k,:) = y(:,1:nb,k,:);
! end;
! 
*/
      for(rocblas_int kr=1; kr <= nblocks; kr++) {
	  rocblas_int k = nblocks - kr + 1;
	  if (k+1 <= nblocks) {
/*
!     ----------------------------------------------------------
!     y(:,1:nb,k,:) = y(:,1:nb,k,:) - U(:,1:nb,1:nb,k) * x(:,1:nb,k+1,:);
!     ----------------------------------------------------------
*/
            

          rocblas_int const iv = 1;
          rocblas_int const mm = nb;
          rocblas_int const nn = nrhs;
          rocblas_int const kk = nb;
          T const alpha = -1;
          T const beta = 1;
          rocblas_int const ld1 = ldu;
          rocblas_int const ld2 = ldx * nblocks;
          rocblas_int const ld3 = ldy * nblocks;

	  rocblas_int const linfo = gemm_nn_vec( nvec, mm,nn,kk,
			  alpha, &(U(iv,1,1,k)),   ld1,
				 &(x(iv,1,k+1,1)), ld2,
                          beta,  &(y(iv,1,k,1)),   ld3 );
	  info = ( (linfo != 0) && (info == 0)) ? linfo : info;
	  };

       };
      return( info );
};
