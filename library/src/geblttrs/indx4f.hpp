	inline auto indx4f = [=]( 
			rocblas_int const i1,
			rocblas_int const i2,
			rocblas_int const i3,
			rocblas_int const i4,  

			rocblas_int const n1,
			rocblas_int const n2,
			rocblas_int const n3,
			rocblas_int const n4) -> size_t const {
		// -----------------------------------------------------------
		// idx = (i1-1) + (i2-1)*n1 + (i3-1)*(n1*n2) + (i4-1)*(n1*n2*n3);
		// or
		// idx = (((i4-1)*n3 + (i3-1))*n2 + (i2-1))*n1 + (i1-1);
		// -----------------------------------------------------------

		assert( (1 <= i1) && (i1 <= n1));
		assert( (1 <= i2) && (i2 <= n2));
		assert( (1 <= i3) && (i3 <= n3));
		assert( (1 <= i4) && (i4 <= n4));

		size_t idx = (i4-1);
		       idx *= n3;
		       idx += (i3-1);
		       idx *= n2;
		       idx += (i2-1);
		       idx *= n1;
		       idx += (i1-1);
		return( idx );
	}
