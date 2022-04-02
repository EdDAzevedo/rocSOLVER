	auto A = [=](   rocblas_int const iv,
			rocblas_int const i,
			rocblas_int const j ) -> T& {
		return( A_[ indx3f( iv,i,j,   nvec,lda,k ] );
		};
