/*
! -------------------------------------------------------------------
! Copyright(c) 2019-2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#include <stdlib.h>
#include <assert.h>
#ifdef USE_CPU


#define GLOBAL_FUNCTION 
typedef int rocblas_int;
typedef int rocblas_status;
typedef void * rocblas_handle;
typedef long rocblas_stride;

#define rocblas_status_success 0

#else
#include "rocblas.hpp"
#define GLOBAL_FUNCTION __global__
#endif


template<typename T>
GLOBAL_FUNCTION
rocblas_status rocsolver_geblttrf_npvt_vec_strided_batched_kernel
( 
rocblas_handle handle,

rocblas_int batchCount,
rocblas_int const ldnvec,

rocblas_int const nb,
rocblas_int const nblocks,

T * A_, 
rocblas_int const lda, 

T * B_, 
rocblas_int const ldb, 

T * C_, 
rocblas_int const ldc,

rocblas_int *pinfo
)
{


#include "geblttrf_npvt_vec.hpp"

#ifdef USE_CPU
	rocblas_int const i_start = 0;
	rocblas_int const i_inc = 1;
	auto const nvec_max = 64;
#else
	rocblas_int const i_start = hipBlockIdx_x;
	rocblas_int const i_inc = hipGridDim_x;
	auto const nvec_max = hipBlockDim_x;
#endif

	auto min = [=]( rocblas_int const m, rocblas_int const n) -> rocblas_int {
		return( (m < n) ? m : n );
	};

	auto const ldnvec = batchCount,
	for(rocblas_int i=i_start; i < batchCount; i += i_inc ) {

		/*
		 * ------------------------------------------------
		 * thread block i performs computation between
		 * ib1 .. (ib2-1), nvec = ib2 - ib1
		 * ------------------------------------------------
		 */
		auto const ib1 =  (i-1) * nvec_max;
		auto const ib2 = min( batchCount, ib1 + nvec_max  );
		auto const nvec = (ib2 - ib1 );

		/*
		 * ---------------------------------------------------------------------
		 * Conceptual view A( 0:(ldnvec-1), 0:(lda-1), 0:(nb-1), 0:(nblocks-1) )
		 * where ldnvec >= batchCount, lda >= nb
		 * ---------------------------------------------------------------------
		 */
		size_t const idxA = (ibatch1-1);
		size_t const idxB = (ibatch1-1);
		size_t const idxC = (ibatch1-1);


		T *Ap = &( A_[idxA] );
		T *Bp = &( B_[idxB] );
		T *Cp = &( C_[idxC] );

		rocblas_int const linfo = gebltrf_npvt_vec( 
				nvec, ldnvec, 
				nb, nblocks,
				Ap, lda,  Bp, ldb, Cp, ldc  );
				
		if ( (linfo != 0) && (info == 0))  {
			info = linfo;
		        };
		};

	*pinfo = info;
};
