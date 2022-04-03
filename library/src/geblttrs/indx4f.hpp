	auto indx4f = [=]( 
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
		rocblas_int constexpr idebug = 0;

		if (idebug >= 1) {
		  assert( (1 <= i1) && (i1 <= n1));
		  assert( (1 <= i2) && (i2 <= n2));
		  assert( (1 <= i3) && (i3 <= n3));
		  assert( (1 <= i4) && (i4 <= n4));
		};

		size_t const ln1 = n1;
		size_t const ln2 = n2;
		size_t const ln3 = n3;
		size_t const idx = (i1-1) + (i2-1)*ln1 + (i3-1)*(ln1*ln2) + 
			           (i4-1)*(ln1*ln2*ln3);

		return( idx );
	};
