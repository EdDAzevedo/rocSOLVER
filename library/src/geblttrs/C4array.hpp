	inline auto C = [=]( 
			rocblas_int const iv,
			rocblas_int const i,
			rocblas_int const j,
			rocblas_int const k ) -> T& {
		return(    C_[indx4f(iv,i,j,k, nvec,ldc,nb,nblocks)] );
	}
