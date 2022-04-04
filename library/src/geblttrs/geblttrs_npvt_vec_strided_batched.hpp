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
rocblas_status rocsolver_geblttrs_npvt_vec_strided_batched_kernel( 
		rocblas_handle handle,
		rocblas_int const nvec,
		rocblas_int const nb,
		rocblas_int const nblocks,
		rocblas_int batchCount_group,
		T * A_, rocblas_int const lda, rocblas_stride const strideA,
		T * B_, rocblas_int const ldb, rocblas_stride const strideB,
		T * C_, rocblas_int const ldc, rocblas_stride const strideC,
		rocblas_int *pinfo
		)
{


#include "geblttrs_npvt_vec.hpp"

	rocblas_int info = atomicMax( pinfo, 0);

#ifdef USE_CPU
	rocblas_int const i_start = 0;
	rocblas_int const i_inc = 1;
	bool const is_root = true;
#else
	rocblas_int const i_start = hipBlockIdx_x;
	rocblas_int const i_inc = hipGridDim_x;
	bool const is_root = (hipThreadIdx_x == 0);
#endif

	for(rocblas_int i=i_start; i < batchCount_group; i += i_inc ) {
		size_t const idxA = i * strideA;
		size_t const idxB = i * strideB;
		size_t const idxC = i * strideC;

		T *Ap = &( A_[idxA] );
		T *Bp = &( B_[idxA] );
		T *Cp = &( C_[idxA] );

		rocblas_int const linfo = rocsolver_geblttrs_npvt_vec( 
				nvec, nb, nblocks,
				Ap, lda,  Bp, ldb, Cp, ldc  );

		if ((linfo != 0) && ( info == 0))  {
			info = atomicMax( pinfo, linfo );
		};
	};


	return( rocblas_success );
};
