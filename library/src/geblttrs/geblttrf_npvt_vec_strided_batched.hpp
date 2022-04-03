/*
! -------------------------------------------------------------------
! Copyright(c) 2019-2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#include <cstddef>
#include <cassert>
#ifdef USE_CPU
typedef int rocblas_int;
typedef int rocblas_status;
typedef void * rocblas_handle;
typedef long rocblas_stride;
#define rocblas_success 0
#else
#include "rocblas.hpp"
#endif


template<typename T>
rocblas_status rocsolver_geblttrf_npvt_vec_strided_batched_kernel( 
		rocblas_handle handle,
		rocblas_int const nvec,
		rocblas_int const nb,
		rocblas_int const nblocks,
		rocblas_int batchCount_group,
		T * A_, rocblas_int const lda, rocblas_stride const strideA,
		T * B_, rocblas_int const ldb, rocblas_stride const strideB,
		T * C_, rocblas_int const ldc, rocblas_stride const strideC,
		rocblas_int info_array[]
		)
{
	rocblas_int info = 0;
	rocblas_int linfo = 0;
	rocblas_status istatus = rocblas_success;



#include "geblttrf_npvt_vec.hpp"

#ifdef USE_CPU
	rocblas_int const i_start = 0;
	rocblas_int const i_inc = 1;
#else
	rocblas_int const i_start = hipThreadIdx_x;
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
				A, lda,  B, ldb, C, ldc  );
				
		};

	return( istatus );
};
