	auto U = [=]( rocblas_int const iv,
		      rocblas_int const i,
		      rocblas_int const j,
		      rocblas_int const k ) -> T& {
		return(    U_[indx4f(iv,i,j,k,      nvec,ldu,nb,nblocks)] );
	};
