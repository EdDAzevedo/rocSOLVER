/*
! -------------------------------------------------------------------
! Copyright(c) 2019-2022. Advanced Micro Devices, Inc. All rights reserved
! -------------------------------------------------------------------
*/

#include "geblttrf_common.hpp"

template<typename T>
GLOBAL_FUNCTION
void geblttrf_npvt_vec_batched_lijk_kernel
( 

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

	rocblas_int info = 0;

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


		T *Ap = &( A_[ ib1 ] );
		T *Bp = &( B_[ ib1 ] );
		T *Cp = &( C_[ ib1 ] );

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


template<typename T>
HOST_FUNCTION 
rocblas_status geblttrf_npvt_vec_batched_lijk
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
rocblas_int const ldc

)
{
	rocblas_int info = 0;
#ifdef USE_CPU
	geblttrf_npvt_vec_batched_lijk_kernel<T>(
			batchCount, ldnvec,
			nb,nblocks,
			A_, lda,
			B_, ldb,
			C_, ldc,
			&info
			);
#else
	nthreads = warpSize;
	ngrid = (batchCount + (nthreads-1))/nthreads;

	geblttrf_npvt_vec_batched_lijk_kernel<T><<< 
		dim3(ngrid),dim3(nthreads),0,istream>>>(
                    batchCount, ldnvec,
		    nb, nblocks,
		    A_, lda,
		    B_, ldb,
		    C_, ldc,
		    &info,
		    );
#endif

	return( (info == 0) ? rocblas_status_success : info );
}


