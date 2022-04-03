	auto B = [=]( rocblas_int const iv,
		      rocblas_int const i,
		      rocblas_int const j,
		      rocblas_int const k ) -> T& {
		return(    B_[indx4f(iv,i,j,k,      nvec,ldb,nb,nblocks)] );
	};
