/*
! -------------------------------------------------------------------
! Copyright(c) 2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#ifndef GBTRF_NPVT_BF_HPP
#define GBTRF_NPVT_BF_HPP

#include "gbtr_common.h"

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
#include "getrf_npvt_bf.hpp"
#include "gemm_nn_bf.hpp"
#include "getrs_npvt_bf.hpp"


template< typename T>
GLOBAL_FUNCTION void
gbtrf_npvt_bf_kernel
(
rocblas_int const nb,
rocblas_int const nblocks,
rocblas_int const batchCount,
T  *              A_,
rocblas_int const lda,
T  *              B_,
rocblas_int const ldb,
T  *              C_,
rocblas_int const ldc,
rocblas_int *pinfo
)
{

#define A(iv,ia,ja,k) A_[ indx4f(iv,ia,ja,k,   batchCount,lda,nb) ]
#define B(iv,ib,jb,k) B_[ indx4f(iv,ib,jb,k,   batchCount,ldb,nb) ]
#define C(iv,ic,jc,k) C_[ indx4f(iv,ic,jc,k,   batchCount,ldc,nb) ]

rocblas_int info = 0;

/*
!     --------------------------
!     reuse storage
!     over-write matrix B with D
!     over-write matrix C with U
!     --------------------------
*/

#define D(iv,i,j,k) B(iv,i,j,k)
#define U(iv,i,j,k) C(iv,i,j,k)
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
! idebug = 1;
! 
! % ----------------------------------
! % in actual code, overwrite B with D
! % overwrite C with U
! % ----------------------------------
! D = zeros(nb,nb,nblocks);
! U = zeros(nb,nb,nblocks);
! 
! use_getrf_npvt = 1;
! 
! k = 1;
! D(:,1:nb,1:nb,k) = B(:,1:nb,1:nb,k);
! if (use_getrf_npvt),
!   D(:,1:nb,1:nb,k) = getrf_npvt( D(:,1:nb,1:nb,k) );
! end;
*/

      {
      rocblas_int const iv = 1;
      rocblas_int const k = 1;
      rocblas_int const mm = nb;
      rocblas_int const nn = nb;
      T *Ap = &(D(iv,1,1,k));
/*
!   ----------------------------------------------
!   D(:,1:nb,1:nb,k) = getrf_npvt( D(:,1:nb,1:nb,k) );
!   ----------------------------------------------
*/
      rocblas_int linfo = 0;
      getrf_npvt_bf_device<T>( batchCount, 
                        mm, 
                        nn, 
                        Ap,
                        ldd, 
                        &linfo );
      SYNCTHREADS;
      info = (linfo != 0) && ( info == 0)  ? (k-1)*nb+linfo : info;
      };

   
/*
! 
! 
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
*/



       for(rocblas_int k=1; k <= (nblocks-1); k++) {
/*
!     --------------------------------------------------------------     
!     U(:,1:nb,1:nb,k) = getrs_npvt( D(:,1:nb,1:nb,k), C(1:nb,1:nb,k) );
!     --------------------------------------------------------------     
*/
         {
         rocblas_int const nn = nb;
         rocblas_int const nrhs = nb;
         rocblas_int const iv = 1;

         rocblas_int linfo = 0;

         T const * const Ap = &(D(iv,1,1,k));
         T       *       Bp = &(C(iv,1,1,k));
         getrs_npvt_bf<T>( batchCount,                                  
                           nn,
                           nrhs, 
                           Ap,
                           ldd,                  
                           Bp,
                           ldc, 
                           &linfo );
          SYNCTHREADS;

         info = (linfo != 0) && (info == 0) ? (k-1)*nb+linfo : info;

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
          T const beta  =  1;
          rocblas_int const ld1 = lda;
          rocblas_int const ld2 = ldu;
          rocblas_int const ld3 = ldd;

          T const * const Ap = &(A(iv,1,1,k+1));
          T const * const Bp = &(U(iv,1,1,k));
          T       *       Cp = &(D(iv,1,1,k+1));
          gemm_nn_bf_device<T>( 
	                 batchCount,                                     
                         mm,
                         nn,
                         kk, 
                         alpha, 
                         Ap,
                         ld1,             
                         Bp,
                         ld2,              
                         beta,  
                         Cp,
                         ld3 );
          SYNCTHREADS;
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
          T *Ap = &(D(iv,1,1,k+1));

          rocblas_int linfo = 0;
          getrf_npvt_bf_device<T>( batchCount,                                 
                            mm,
                            nn,
                            Ap,
                            ldd, 
                            &linfo );

          SYNCTHREADS;

          info = (linfo != 0) && (info == 0) ? (k-1)*nb+linfo : info;
          };


      }; // end for k

  if (info != 0) {
     *pinfo = info;
     };
}
#undef D
#undef U

#undef A
#undef B
#undef C



template< typename T>
void gbtrf_npvt_bf_template
(
hipStream_t stream,

rocblas_int const nb,
rocblas_int const nblocks,
rocblas_int const batchCount,
T  *              A_,
rocblas_int const lda,
T  *              B_,
rocblas_int const ldb,
T  *              C_,
rocblas_int const ldc,
rocblas_int *pinfo
)
{

#ifdef USE_GPU
  int block_dim = 64;
  int grid_dim = (batchCount + (block_dim-1))/block_dim;
  hipLaunchKernelGGL( (gbtrf_npvt_bf_kernel<T>),
                      dim3( grid_dim ),
                      dim3( block_dim ),
                      0,
                      stream,

                      nb,
                      nblocks,
                      batchCount,
                      A_,
                      lda,
                      B_,
                      ldb,
                      C_,
                      ldc,
                      pinfo 
                   );
#else

     gbtrf_npvt_bf_kernel<T>(
                      nb,
                      nblocks,
                      batchCount,
                      A_,
                      lda,
                      B_,
                      ldb,
                      C_,
                      ldc,
                      pinfo 
                      );

#endif
}


#endif
