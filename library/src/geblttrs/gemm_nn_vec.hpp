template<classname T>
auto gemm_nn_vec = [=](
		rocblas_int const nvec,
		rocblas_int const m,
		rocblas_int const n,
		rocblas_int const k,
		T const alpha,
		T *A_, rocblas_int lda,
		T *B_, rocblas_int ldb,
		T const beta,
		T *C_, rocblas_int ldc ) -> rocblas_int {
/*
 * --------------------------------------------
 * Perform GEMM operation within a thread block.
 * equivalent to
 * do iv=1,nvec
 *    C(iv, 1:m,1:n) = beta * C(iv,1:m,1:n) +  &
 *         alpha * matmul( A(iv,1:m,1:k), B(iv,1:k,1:n))
 * --------------------------------------------
*/
#include "indx3f.hpp"







	rocblas_int const info = 0;

#ifdef  USE_CPU
	rocblas_int const iv_start = 1;
	rocblas_int const iv_inc = 1;
#else
	rocblas_int const iv_start = hipThreadId_x;
	rocblas_int const iv_inc = hipThreadDim_x;
	__syncthreads();
#endif

	for(rocblas_int jc=1; jc <= n; jc++) {
        for(rocblas_int ic=1; ic <= m; ic++) {

         for(rocblas_int iv=iv_start; iv <= iv_end; iv += iv_inc) {

		T cij = 0;

                #pragma unroll
		for(rocblas_int ja=1; ja <= k; ja++) {
			cij += A(iv,ic,ja) * B(iv,ja,jc);
		};

		if (beta == 0) {
	           C(iv,ic,jc) = alpha * cij;
		}
		else {
			C(iv,ic,jc) = beta*C(iv,ic,jc) + alpha * cij;
		};

	  };
	};
	};

#ifndef USE_CPU
	__syncthreads();
#endif
        return( info );
}

