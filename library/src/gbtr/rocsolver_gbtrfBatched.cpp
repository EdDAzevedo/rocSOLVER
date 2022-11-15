
/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "rocsolver_gbtrfBatched.hpp"

template<typename T, typename I>
rocblas_status rocsolver_gbtrfBatched_impl(
	rocblas_handle handle,
	I nb,
	I nblocks,
	T* A_array[],
	I lda,
	T* B_array[],
	I ldb,
	T* C_array[],
	I ldc,
	I batchCount
	) 
{
  
  hipStream_t stream;
  rocblas_get_stream( handle, &stream );

  I host_info = 0;
  gbtrf_npvt_batched_template<T,I>(stream, nb, nblocks, batchCount, A_array, lda, B_array, ldb,
                                        C_array, ldc, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
}


extern "C" {

rocblas_status rocsolver_dgbtrfBatched(rocblas_handle handle,
                                      rocblas_int nb,
                                      rocblas_int nblocks,
                                      double* A_array[],
                                      rocblas_int lda,
                                      double* B_array[],
                                      rocblas_int ldb,
                                      double* C_array[],
                                      rocblas_int ldc,
                                      rocblas_int batchCount)
{

 return( rocsolver_gbtrfBatched_impl<double,rocblas_int>(
                            handle, nb, nblocks,
                            A_array, lda,
                            B_array, ldb,
                            C_array, ldc,
                            batchCount ) );
};




rocblas_status rocsolver_sgbtrfBatched(rocblas_handle handle,
                                      rocblas_int nb,
                                      rocblas_int nblocks,
                                      float* A_array[],
                                      rocblas_int lda,
                                      float* B_array[],
                                      rocblas_int ldb,
                                      float* C_array[],
                                      rocblas_int ldc,
                                      rocblas_int batchCount)
{

 return( rocsolver_gbtrfBatched_impl<float,rocblas_int>(
                            handle, nb, nblocks,
                            A_array, lda,
                            B_array, ldb,
                            C_array, ldc,
                            batchCount ) );
};



rocblas_status rocsolver_zgbtrfBatched(rocblas_handle handle,
                                      rocblas_int nb,
                                      rocblas_int nblocks,
                                      rocblas_double_complex* A_array[],
                                      rocblas_int lda,
                                      rocblas_double_complex* B_array[],
                                      rocblas_int ldb,
                                      rocblas_double_complex* C_array[],
                                      rocblas_int ldc,
                                      rocblas_int batchCount)
{

 return( rocsolver_gbtrfBatched_impl<rocblas_double_complex,rocblas_int>(
                            handle, nb, nblocks,
                            A_array, lda,
                            B_array, ldb,
                            C_array, ldc,
                            batchCount ) );
};



rocblas_status rocsolver_cgbtrfBatched(rocblas_handle handle,
                                      rocblas_int nb,
                                      rocblas_int nblocks,
                                      rocblas_float_complex* A_array[],
                                      rocblas_int lda,
                                      rocblas_float_complex* B_array[],
                                      rocblas_int ldb,
                                      rocblas_float_complex* C_array[],
                                      rocblas_int ldc,
                                      rocblas_int batchCount)
{

 return( rocsolver_gbtrfBatched_impl<rocblas_float_complex,rocblas_int>(
                            handle, nb, nblocks,
                            A_array, lda,
                            B_array, ldb,
                            C_array, ldc,
                            batchCount ) );
};

}
