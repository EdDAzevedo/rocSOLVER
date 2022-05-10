auto gemm_nn_vec = [=](
		rocblas_int const nvec,
		rocblas_int const ldnvec,

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
	rocblas_int const ncolA = k;
	rocblas_int const ncolB = n;
	rocblas_int const ncolC = n;

#include "indx3f.hpp"
#include "A3array.hpp"
#include "B3array.hpp"
#include "C3array.hpp"






	rocblas_int const info = 0;

	rocblas_int const iv_end = nvec;
#ifdef  USE_CPU
	rocblas_int const iv_start = 1;
	rocblas_int const iv_inc = 1;
#else
	rocblas_int const iv_start = 1 + hipThreadIdx_x;
	rocblas_int const iv_inc = hipBlockDim_x;
#endif

        for(auto iv=iv_start; iv <= iv_end; iv += iv_inc) {
	for(auto jc=1; jc <= n; jc++) {
        for(auto ic=1; ic <= m; ic++) {


		T cij = 0;

		rocblas_int const ja = 1;
		T const * const Ap = &(A(iv,ic,ja));
		T const * const Ap_inc = &(A(iv,ic,ja+1)) - Ap;

		T const * const Bp = &(B(iv,ja,jc));
		T const * const Bp_inc = &(B(iv,ja+1,jc)) - Bp;

		for(rocblas_int ja=1; ja <= k; ja++) {
			// cij += A(iv,ic,ja) * B(iv,ja,jc);
			T const Aik = *Ap;
			T const Bkj = *Bp;
			cij += Aik * Bkj;
			Ap += Ap_inc;
			Bp += Bp_inc;
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

	SYNCTHREADS;

        return( info );
};

