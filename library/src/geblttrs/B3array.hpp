	auto B = [=](   rocblas_int const iv,
			rocblas_int const i,
			rocblas_int const j ) -> T& {
		return( B_[ indx3f( iv,i,j,   ldnvec,ldb,ncolB) ] );
		};
