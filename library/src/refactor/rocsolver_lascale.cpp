
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
#include "rocblas/rocblas.h"
#include "rocsolver_lascale.hpp"

template <typename T>
rocblas_status rocsolver_lascale_impl(
	rocblas_handle handle,
	const rocblas_int nrow,
	const rocblas_int ncol,
	T const * const drow,
	T const * const dcol,
	rocblas_int const * const Ap,
	rocblas_int const * const Ai,
	T                 * const Ax
        )
{
   hipStream_t stream;
   rocblas_get_stream( handle, &stream);

   rocsolver_scale_template(
	stream,
	nrow,
	ncol,
	drow,
	dcol,
	Ap,
	Ai,
	Ax
        );
   return( rocblas_status_success );
}

extern "C" {

rocsolverStatus_t rocsolverDlascale(
	rocblas_handle handle,
	const rocblas_int nrow,
	const rocblas_int ncol,
	double const * const drow,
	double const * const dcol,
	rocblas_int const * const Ap,
	rocblas_int const * const Ai,
	double            * const Ax
        )
{
   return( rocsolver_lascale_impl<double>(
                handle,
                nrow,
                ncol,
                drow,
                dcol,
                Ap,
                Ai,
                Ax
                ) );
}



rocsolverStatus_t rocsolverSlascale(
	rocblas_handle handle,
	const rocblas_int nrow,
	const rocblas_int ncol,
	float const * const drow,
	float const * const dcol,
	rocblas_int const * const Ap,
	rocblas_int const * const Ai,
	float            * const Ax
        )
{
   return( rocsolver_lascale_impl<float>(
                handle,
                nrow,
                ncol,
                drow,
                dcol,
                Ap,
                Ai,
                Ax
                ) );
}


rocsolverStatus_t rocsolverZlascale(
	rocblas_handle handle,
	const rocblas_int nrow,
	const rocblas_int ncol,
	rocblas_double_complex const * const drow,
	rocblas_double_complex const * const dcol,
	rocblas_int const * const Ap,
	rocblas_int const * const Ai,
	rocblas_double_complex            * const Ax
        )
{
   return( rocsolver_lascale_impl<rocblas_double_complex>(
                handle,
                nrow,
                ncol,
                drow,
                dcol,
                Ap,
                Ai,
                Ax
                ) );
}



rocsolverStatus_t rocsolverZlascale(
	rocblas_handle handle,
	const rocblas_int nrow,
	const rocblas_int ncol,
	rocblas_float_complex const * const drow,
	rocblas_float_complex const * const dcol,
	rocblas_int const * const Ap,
	rocblas_int const * const Ai,
	rocblas_float_complex            * const Ax
        )
{
   return( rocsolver_lascale_impl<rocblas_float_complex>(
                handle,
                nrow,
                ncol,
                drow,
                dcol,
                Ap,
                Ai,
                Ax
                ) );
}






}
