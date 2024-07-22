
/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.9.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     November 2019
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "auxiliary/rocauxiliary_lacgv.hpp"
#include "auxiliary/rocauxiliary_larf.hpp"
#include "auxiliary/rocauxiliary_larfg.hpp"
#include "roclapack_geqr2.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

template <bool BATCHED, typename T>
static 
void rocsolver_rgeqr3_getMemorySize(const rocblas_int m,
                                   const rocblas_int n,
                                   const rocblas_int batch_count,
                                   size_t* size_T,
                                   size_t* size_W)
{
        *size_T = 0;
	*size_W = 0;
    // if quick return no workspace needed
    if(m == 0 || n == 0 || batch_count == 0)
    {
        return;
    }

    const size_t nb = GEQxF_BLOCKSIZE;
    *size_T = sizeof(T) * nb * nb * std::max(1,batch_count);
    *size_W = sizeof(T) * nb * nb * std::max(1,batch_count);

}

static
double dconj( double x ) { return( x ); };
static
float dconj( float x) { return(x); };
static
std::complex<float> dconj( std::complex<float> x ) {
	return( std::conj(x) ); };
static
std::complex<double> dconj( std::complex<double> x) {
	return( std::conj(x) ); };
static
rocblas_complex_num<float> dconj( rocblas_complex_num<float> x ) {
	return( conj( x ) );
};

static
rocblas_complex_num<double> dconj( rocblas_complex_num<double> x ) {
	return( conj( x ) );
};



// -----------------------------------------
// geadd() performs matrix addition that is
// similar PxGEADD() in Parallel BLAS library
//
//  C(1:m,1:n) =  beta * C(1:m,1:n) + alpha * op(A)
//
// assume launch with 
// dim3(1,1,max_nblocks), dim3(waprSize,1024/warpSize)
// -----------------------------------------
template<typename T, typename UA, typename UC, typename I, typename Istride>
static
void geadd_kernel( char trans, 
		I const m, I const n, I const batch_count,
		T const alpha, 
		UA AA, I const shiftA, I const ldA, Istride const strideA,
		T const beta,
		UC CC, I const shiftC, I const ldC, Istride const strideC )
{

  bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
  if (!has_work) { return; };


  auto const i_start = hipThreadIdx_x;
  auto const i_inc = hipBlockDim_x;
  auto const j_start = hipThreadIdx_y;
  auto const j_inc = hipBlockDim_y;

  bool const is_transpose = (trans == 'T' || trans == 't');
  bool const is_conj_transpose = (trans == 'C' || trans == 'c');

  auto const bid_inc = hipGridDim_z;
  auto const bid_start = hipBlockIdx_z;

  T const zero = 0;

  for(I bid = bid_start; bid < batch_count; bid += bid_inc) {

	T const * const A = load_ptr_batch( AA, bid, shiftA, strideA );
	T       * const C = load_ptr_batch( CC, bid, shiftC, strideC );

	for(auto j=j_start; j < n; j += j_inc) {
        for(auto i=i_start; i < m; i += i_inc) {
		auto const ij = idx2D(i,j,ldC );
		C[ ij ] = (beta == zero) ? zero : beta * C[ ij ];


		T const aij = (is_transpose) ? A[ idx2D(j,i,ldA) ] :
			      (is_conj_transpose) ? dconj( A[ idx2D(j,i,ldA) ] ) :
			      A[ idx2D(i,j,ldA) ];


		C[ ij ] += alpha * aij;
	}
	}
  }

}

template<typename T, typename UA, typename UC, typename I, typename Istride>
static
void geadd_template( char trans, 
		I const m, I const n, I const batch_count,
		T const alpha, 
		UA AA, I const shiftA, I const ldA, Istride const strideA,
		T const beta,
		UC CC, I const shiftC, I const ldC, Istride const strideC )
{

	auto const max_threads = 1024;
	auto const max_blocks = 64*1000;

	auto const nblocks = std::min( max_blocks, batch_count );
	auto const nx = warpSize;
	auto const ny = max_threads/nx;

	geadd_kernel<T,UA,UC,I,Istride><<< dim3(1,1,nblocks), dim3(nx,ny,1) >>>( 
			trans, m,n,batch_count,
			alpha,  
			AA, shiftA, ldA, strideA,
			beta,
			CC, shiftC, ldC, strideC );

}

template<typename T,  typename U, typename I, typename Istride >
static
rocblas_status formT3( I const m, I const k1, I const k2,  
		       I const batch_count,

		       U Y, I const shiftY, I const ldY, Istride strideY,
		    T* const Tmat, 
		    I  const shiftT1, 
		    I  const shiftT2,
		    I  const shiftT3,
		    Istride const strideT, 
		    T* const Wmat, 
		    I const ldW, 
		    Istride const strideW ) 
{
        auto const k = k1 + k2;

	auto const shiftY11 = idx2D(0,0,ldY);
	auto const shiftY21 = idx2D(k1,0,ldY);
	auto const shiftY31 = idx2D(k,0,ldy);

	auto const shiftY12 = idx2D(k1,k1,ldY);
	auto const shiftY22 = idx2D(k,k1,ldY);

 // -------------------------------
 //  [W] = formT3( Y1, T1, Y2, T2 )
 //  
 //  compute W = -T1 * (Y1' * Y2 ) * T2
 // 
 //  Y1 is m by k1
 //  Y2 is (m-k1) by k2
 // 
 //  Let
 //  Y1 = [Y11; Y21; Y31];
 // 
 //  Y1 = [ Y11 ]
 //       [ Y21 ]
 //       [ Y31 ]
 // 
 //  Y11 is k1 by k1 unit lower diagonal but not used
 //  Y21 is k2 by k1
 //  Y31 = (m-k) by k1,   where k = k1 + k2
 // 
 //  Y2 = [   0 ]
 //       [ Y12 ]
 //       [ Y22 ]
 //
 //  Y12 is k2 by k2 unit lower diagonal
 //  Y22 is (m-k) by k2
 // -------------------------------




 auto const nrow_Y21 = k2;
 auto const ncol_Y21 = k1;

 auto const nrow_Y31 = (m-k);
 auto const ncol_Y31 = k1;
 
 auto const nrow_Y12 = k2;
 auto const ncol_Y12 = k2;

 auto const nrow_Y22 = (m-k);
 auto const ncol_Y22 = k2;

 auto const shiftY11 = idx2D( 0,0, ldY1 );
 auto const shiftY21 = idx2D( k1,0,ldY1 );
 auto const shiftY31 = idx2D( k1+k2,0,ldY1);

 auto const shiftY12 = idx2D( k1,k1,ldY2);
 auto const shiftY22 = idx2D( k1+k2,k1,ldY2 );

// ------------------------
// (0) first set W = Y21'
// (1) TRMM   W = Y21' * Y12
// (2) GEMM   W = W + Y31' * Y22
// (3) TRMM   W = -T1 * W
// (4) TRMM   W = W * T2
// ------------------------

 {
 // ---------------------
 // (0) first set W = Y21'
 // ---------------------
 char const trans = 'C';
 I const mm = ncols_Y21;
 I const nn = nrows_Y21;
 T const alpha = 1;
 T const beta = 0;

 Istride const strideW = 
 I const shiftW = 0;
 geadd_template( trans, mm,nn,batch_count,
		 alpha,
		 Y1, shiftY1 + shiftY21, ldY1, strideY1,
		 beta,
		 Wmat, shiftW, ldW, strideW );
 }
 auto const nrow_W = ncol_Y21;
 auto const ncol_W = nrow_Y21;

 {
 // --------------------------------------
 // (1) TRMM   W = W * Y12, where W = Y21'
 // --------------------------------------
 rocblas_side const side =  rocblas_side_right;
 rocblas_fill const uplo = rocblas_fill_lower;
 rocblas_operation const transA =  rocblas_operation_none;
 rocblas_diagonal const diag =  rocblas_diagonal_unit;
 I const mm =  nrow_W;
 I const nn =  ncol_W;

 const T* alpha =  &(alpha_array[0]);
 Istride const stride_alpha =  1;
 T const * const A = 
 Istride const offsetA = 
 I const lda = 
 Istride const strideA = 

 T* const B = 
 Istride const offsetB = 
 I const ldb = 
 Istride const strideB = 
 I const batch_count = 
 HIP_CHECK( rocblasCall_trmm(handle,
                                rocblas_side side,
                                rocblas_fill uplo,
                                rocblas_operation transA,
                                rocblas_diagonal diag,
                                rocblas_int m,
                                rocblas_int n,
                                const T* alpha,
                                rocblas_stride stride_alpha,
                                const T* const* A,
                                rocblas_stride offsetA,
                                rocblas_int lda,
                                rocblas_stride strideA,
                                T* B,
                                rocblas_stride offsetB,
                                rocblas_int ldb,
                                rocblas_stride strideB,
                                rocblas_int batch_count
                                ) );
 }




}

// --------------------------------------------------
// Algorithm inspired by the paper
// "Applying recursion to serial and parallel QR factorization"
// by Erik Elmroth and Fred G. Gustavson
// IBM Journal of Research and Development, August 2000
//
//
//  Input A(0:(m-1), 0:(n-1))
//  Output (Y, R, T),  Y and R replaces A, m >= n
//
//  R is n by n upper triangular 
//  Y is lower trapezoidal with ones's on the diagonal
//  or Y(i,i) == 1, for i=0:(n-1)
//
//  Note T is n by n  upper triangular matrix
//  The diagonal entries T(i,i) are the "tau" values
//  in lapack GEQRF
// --------------------------------------------------

template <typename T, typename U, typename I, typename Istride, bool COMPLEX = rocblas_is_complex<T>>
static 
rocblas_status rocsolver_rgeqr3_template(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        U A,
                                        const I shiftA,
                                        const I lda,
                                        const Istride strideA,
                                        const I batch_count,
					T* const Tmat, I const ldT,
					T* const Wmat, I const ldW,
	                                T* work_workArr, T* Abyx_norms	)
{
    ROCSOLVER_ENTER("rgeqr3", "m:", m, "n:", n, "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);



    if (n == 1) {
	// ------------------------------------------------------
        // generate Householder reflector to work on first column
	// ------------------------------------------------------
        const I j = 0;
	Istride const strideT = (ldT * n);

        rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(j, j, lda), A,
                                 shiftA + idx2D(std::min(j + 1, m - 1), j, lda), 1, strideA,
                                 Tmat, strideT, batch_count, (T*)work_workArr, Abyx_norms);

    }
    else {
	    auto const n1 = n/2;
	    auto const n2 = n - n1;

	    // --------------------------------------------
	    // [Y1, R1, T1 ] = rgeqr3( A(0:(m-1), 0:(n1-1))  
	    // --------------------------------------------

	    T* const T1 = Tmat;
            HIP_CHECK( rocsolver_rgeqr3_template(handle,
                                        m,
                                        k1,
                                        A,
                                        shiftA,
                                        lda,
                                        strideA,
                                        batch_count,
	                                work_workArr, 
					Abyx_norms,	 
					T1, 
					ldT,
					Wmat, 
					ldW)  );

	    // -----------------------------------------------------
	    // compute  A(0:(m-1), n1:n ) = Q1' * A( 0:(m-1), n1:n )
	    // -----------------------------------------------------


	    auto const shiftY1 = idx2D(0,0,ldA);
	    auto const shiftA2 = idx2D(0,n1,ldA);
	    HIP_CHECK( applyQtC( stream, m, n2, n1,
				    A + shiftA + shiftY1, lda, strideA, // Y1
				    Tmat, ldT,                          // T1
				    A + shiftA + shiftA2, lda, strideA ) );

				    
									



	    // -----------------------------------------
	    // [Y2, R2, T2 ] = rgeqr3( A( n1:m, n1:n ) )
	    // -----------------------------------------

            T* const T2 = Tmat + idx2D( n1,n1, ldT );

            HIP_CHECK( rocsolver_rgeqr3_template(handle,
                                        m,
                                        n2,
                                        A, shiftA + shiftA2, lda, strideA,
                                        batch_count,
	                                work_workArr, 
					Abyx_norms,	 
					T2, ldT,
					Wmat, ldW)  );

	    // -------------------------------------------------------
	    // compute T3 = T(0:(n1-1), n1:n ) = -T1 * (Y1' * Y2) * T2
	    // -------------------------------------------------------

	    // ------------------------------------
	    // Note that 
	    // Y1 is m by n1 unit lower trapezoidal,  
	    // Y2 is (m-n1) by n2 lower trapezoidal
	    // ------------------------------------
	    T* const T3 = Tmat + idx2D( 0, n1, ldT );
	    HIP_CHECK( formT3( m, n1, n2,  
				    A, shiftA + shiftY1, lda, strideA, // Y1
				    A, shiftA + shiftY2, lda, strideA, // Y2
				    T1, ldT,
				    T2, ldT,
			            T3, ldT,
			            Wmat, ldW );

	    // --------------------------------------------------------------
	    // implicitly form Y = [Y1, Y2] where Y is unit lower trapezoidal
	    // Note Y over-writes lower part of A
	    // --------------------------------------------------------------
	    //

	    // -----------------------------------
	    // R = [ R1     A(0:(n1-1), n1:(n-1)) ]
	    //     [ 0      R2                    ]
	    //
	    // Note R is n by n upper triangular
	    // and over-writes matrix A
	    // -----------------------------------

	    // -----------------------------------
	    // T = [ T1     T3 ]
	    //     [ 0      T2 ]
	    // -----------------------------------


    }
    return( rocblas_status_success );
}




    for(rocblas_int j = 0; j < dim; ++j)
    {
        // generate Householder reflector to work on column j
        rocsolver_larfg_template(handle, m - j, A, shiftA + idx2D(j, j, lda), A,
                                 shiftA + idx2D(std::min(j + 1, m - 1), j, lda), 1, strideA,
                                 (ipiv + j), strideP, batch_count, (T*)work_workArr, Abyx_norms);

        // insert one in A(j,j) tobuild/apply the householder matrix
        ROCSOLVER_LAUNCH_KERNEL(set_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream,
                                diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda, strideA, 1, true);

        // conjugate tau
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, 1, ipiv, j, 1, strideP, batch_count);

        // Apply Householder reflector to the rest of matrix from the left
        if(j < n - 1)
        {
            rocsolver_larf_template(handle, rocblas_side_left, m - j, n - j - 1, A,
                                    shiftA + idx2D(j, j, lda), 1, strideA, (ipiv + j), strideP, A,
                                    shiftA + idx2D(j, j + 1, lda), lda, strideA, batch_count,
                                    scalars, Abyx_norms, (T**)work_workArr);
        }

        // restore original value of A(j,j)
        ROCSOLVER_LAUNCH_KERNEL(restore_diag<T>, dim3(batch_count, 1, 1), dim3(1, 1, 1), 0, stream,
                                diag, 0, 1, A, shiftA + idx2D(j, j, lda), lda, strideA, 1);

        // restore tau
        if(COMPLEX)
            rocsolver_lacgv_template<T>(handle, 1, ipiv, j, 1, strideP, batch_count);
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
