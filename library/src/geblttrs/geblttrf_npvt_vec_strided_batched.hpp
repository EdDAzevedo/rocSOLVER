! -------------------------------------------------------------------
! Copyright(c) 2019-2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
#pragma once

template<classname T>
rocblas_status rocsolver_gebltrf_npvt_vec_strided_batched_kernel( 
		rocblas_handle handle,
		roblas_int const nvec,
		rocblas_int const nb,
		rocblas_int const nblocks,
		rocblas_int const batchCount_group,
		T * A_, rocblas_int const lda, rocblas_stride const strideA,
		T * B_, rocblas_int const ldb, rocblas_stride const strideB,
		T * C_, rocblas_int const ldc, rocblas_stride const strideC,
		rocblas_int info_array[],
		rocblas_int batchCount_group
		)
{
	rocblas_int info = 0;
	rocblas_int linfo = 0;
	rocblas_status istatus = rocblas_success;



#include "gebltrf_npvt_vec.hpp"

#ifdef USE_CPU
	rocblas_int const i_start = 0;
	rocblas_int const i_inc = 1;
#else
	rocblas_int const i_start = hipBlockIdx_x;
	rocblas_int const i_inc = hipBlockDim_x;
#endif

	for(rocblas_int i=i_start; i < batchCount_group; i += i_inc ) {
		size_t const idxA = i * strideA;
		size_t const idxB = i * strideB;
		size_t const idxC = i * strideC;

		T *A = &( A[idxA] );
		T *B = &( B[idxA] );
		T *C = &( C[idxA] );

		info_array[i] = rocsolver_gebltrf_npvt_vec( nvec, nb, nblocks,
				A, lda,  B, ldb, C, ldc  )
				
		};

	return( istatus );
}
         


 

      subroutine gbtrf_npvt_vec_strided_batched( nb,nblocks,             &
     &     batchCount_group,                                             &
     &     A,lda, strideA, B, ldb, strideB,                              &
     &     C, ldc, strideC,info,use_gpu_in)
      use precision_mod
      implicit none
      integer, intent(in) :: nb, nblocks, batchCount_group
      integer, intent(in) :: lda,ldb,ldc
      integer(kind=i8), intent(in) :: strideA, strideB, strideC
      real(kind=wp), intent(inout) :: A(strideA*batchCount_group)
      real(kind=wp), intent(inout) :: B(strideB*batchCount_group)
      real(kind=wp), intent(inout) :: C(strideC*batchCount_group)
      integer, intent(inout) :: info
      logical, intent(in) :: use_gpu_in

      integer :: linfo(batchCount_group)
      integer(kind=i8) :: idxA,idxB,idxC
      integer :: i
      logical :: use_gpu

      info = 0
      linfo = 0
      use_gpu = use_gpu_in


#ifdef USE_GPU

#ifdef _OPENACC

!$acc  parallel loop gang                                                    &
!$acc& if (use_gpu)                                                          &
!$acc& num_gangs( batchCount_group )                                         &
!$acc& num_workers(1) vector_length(max(64,min(nvec,1024)))                  &
!$acc& copyin(nb,nblocks,batchCount_group,lda,ldb,ldc)                       &
!$acc& copyin(strideA,strideB,strideC)                                       &
!$acc& copy(A,B,C)                                                           &
!$acc& private(idxA,idxB,idxC)                                               &
!$acc& copy(linfo)
#else

!$omp  target data                                                           &
!$omp& if (use_gpu)                                                          &
!$omp& map(to:nb,nblocks,batchCount_group,lda,ldb,ldc)                       &
!$omp& map(to:strideA,strideB,strideC)                                       &
!$omp& map(A,B,C)                                                            &
!$omp& map(linfo)

!$omp  target teams distribute                                               &
!$omp& if (use_gpu)                                                          &
!$omp& default(none)                                                         &
!$omp& shared(nb,nblocks,batchCount_group,lda,ldb,ldc)                       &
!$omp& shared(strideA,strideB,strideC)                                       &
!$omp& shared(A,B,C)                                                         &
!$omp& private(idxA,idxB,idxC)                                               &
!$omp& shared(linfo)
#endif

#else
!$omp  parallel do private(idxA,idxB,idxC)                                     
#endif
      do i=1,batchCount_group

        idxA = strideA * (i-1) + 1
        idxB = strideB * (i-1) + 1
        idxC = strideC * (i-1) + 1


        call gbtrf_npvt_vec(nb,nblocks,                                  &
     &                  A(idxA),lda,                                     &
     &                  B(idxB),ldb,                                     &
     &                  C(idxC),ldc, linfo(i))
      enddo
#ifdef USE_GPU
#ifdef _OPENACC
#else
!$omp end target data
#endif
#endif

      info = maxval(linfo)

      return
      end subroutine gbtrf_npvt_vec_strided_batched
