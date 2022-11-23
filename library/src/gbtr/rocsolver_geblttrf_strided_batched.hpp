/* ************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "roclapack_getrf.hpp"

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

template<typename T >
rocblas_status rocsolver_geblttrf_strided_batched_large_impl(
	rocblas_handle handle,
	const rocblas_int nb,
	const rocblas_int nblocks,
        T* A_, 
        const rocblas_int lda,
        const rocblas_stride strideA,
        T* B_,
        const rocblas_int ldb,
        const rocblas_stride strideB,
        T* C_,
        const rocblas_int ldc,
        const rocblas_stride strideC,
        rocblas_int info_array[], // array of batch_count integers on GPU
        const rocblas_int batch_count )
{

/*
 -----------------
 arrays dimensioned as 

 A(lda,nb,nblocks, batch_count)
 B(ldb,nb,nblocks, batch_count)
 C(ldc,nb,nblocks, batch_count)
 -----------------
*/

/*
 i1 + i2*n1 + i3*(n1*n2) + i4*(n1*n2*n3)
 or
 ((i4*n3 + i3)*n2 + i2)*n1 + i1
 */
#define indx4(i1,i2,i3,i4, n1,n2,n3,n4) \
    ((((i4)*(n3)+(i3))*(n2)+(i2))*(n1)+(i1))
 
#define indx4f(i1,i2,i3,i4, n1,n2,n3,n4) \
         indx4( ((i1)-1), ((i2)-1), ((i3)-1), ((i4)-1),  n1,n2,n3,n4)

#define A(i1,i2,i3,i4)  \
	A_[ indx4f(i1,i2,i3,i4,   lda,nb,nblocks,batch_count) ]

#define B(i1,i2,i3,i4 ) \
	B_[ indx4f(i1,i2,i3,i4,   ldb,nb,nblocks,batch_count) ]

#define C(i1,i2,i3,i4)  \
	C_[ indx4f(i1,i2,i3,i4,   ldc,nb,nblocks,batch_count) ]


/*
!     --------------------------
!     reuse storage
!     over-write matrix B with D
!     over-write matrix C with U
!     --------------------------
*/
#define D(i1,i2,i3,i4) B(i1,i2,i3,i4)
#define U(i1,i2,i3,i4) C(i1,i2,i3,i4)
rocblas_int const ldd = ldb;
rocblas_stride const strideD = strideB;

rocblas_int const ldu = ldc;
rocblas_stride const strideU = strideC;

 
 





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
 rocblas_int const k = 1;
 rocblas_int const mm = nb;
 rocblas_int const nn = nb;
 rocblas_int const ld1 = ldd;
 rocblas_stride const stride1 = strideD;

 rocblas_int const i = 1;
 rocblas_int const j = 1;
 rocblas_int const ibatch = 1;
 T* Ap = &(D(i,j,k,ibatch));
 rocblas_status istat = rocsolver_getrf_npvt_strided_batched( 
       handle, mm,nn,Ap,ld1,stride1, info_array, batch_count );
 if (istat != rocblas_status_success) {
   return( rocblas_status );
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
  for(rocblas_int k=1; k <= (nblocks-1); k++) {

     {
/*
     -------------------------------------------------------------
     U(1:nb,1:nb,k) = getrs_npvt( D(1:nb,1:nb,k), C(1:nb,1:nb,k) );

     Note reuse storage for C(:,:,:,:) as U(:,:,:,:)
     -------------------------------------------------------------
*/

 
      rocblas_int const nn = nb;
      rocblas_int const nrhs = nb;
        
      rocblas_int const ibatch = 1;
      T const * const Ap = &(D(1,1,k,ibatch));
      rocblas_int const ld1 = ldd;
      rocblas_stride const stride1 = strideD;

      T * const Bp = &(U(1,1,k,ibatch));
      rocblas_int const ld2 = ldu;
      rocblas_stride const stride2 = strideU;

     
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
    T alpha = -1;
    T beta = 1;
    rocblas_int const ibatch = 1;

    rocblas_int const mm = nb;
    rocblas_int const nn = nb;
    rocblas_int const kk = nb;

    T const * const Ap = &(A(1,1,k+1,ibatch));
    rocblas_int const ld1 = lda;
    rocblas_stride const stride1 = strideA;

    T const * const Bp = &(U(1,1,k,ibatch));
    rocblas_int const ld2 = ldu;
    rocblas_int const stride2 = strideU;

    T* const Cp = &(B(1,1,k+1,ibatch));
    rocblas_int const ld3 = ldb;
    rocblas_stride const stride3 = strideB;


    rocblas_status istat = rocsolver_gemm_strided_batched(
          handle, mm,nn,kk, 
          alpha,  Ap, ld1, stride1,
                  Bp, ld2, stride2,
          beta,   Cp, ld3, stride3
          );
    if (istat != rocblas_success) {
       return( istat );
       };
    };


/*
!      --------------------------------------------------
!      D(1:nb,1:nb,k+1) = getrf_npvt( D(1:nb,1:nb,k+1) );
!      --------------------------------------------------
*/
    {

     rocblas_int const mm = nb;
     rocblas_int const nn = nb;

     rocblas_int const ibatch = 1;
     T* const Ap = &(D(1,1,k+1,ibatch));
     rocblas_int const ld1 = ldd;
     rocblas_stride const stride1 = strideD;

     rocblas_status istat = rocsolver_getrf_npvt_strided_batched( 
            handle, mm,nn,
            Ap,ld1,stride1, 
            info_array, batch_count );
     if (istat != rocblas_status_success) {
        return( istat );
        };

     };

  }; // end for k

};







#undef indx4
#undef indx4f
#undef A
#undef B
#undef C
#undef D
#undef U
