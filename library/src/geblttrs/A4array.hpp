	inline auto A = [=]( rocblas_int const iv,
			rocblas_int const i,
			rocblas_int const j,
			rocblas_int const k ) -> T& {
		return(    A_[indx4f(iv,i,j,k, nvec,lda,nb,nblocks)] );
	}
