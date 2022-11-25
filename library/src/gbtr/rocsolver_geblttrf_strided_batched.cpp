
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
#include "rocsolver_geblttrf_strided_batched_small.hpp"
#include "rocsolver_geblttrf_strided_batched_large.hpp"

template <typename T, typename I, typename Istride>
rocblas_status rocsolver_geblttrf_strided_batched_impl(rocblas_handle handle,
                                                       I nb,
                                                       I nblocks,
                                                       T* A_,
                                                       I lda,
                                                       Istride strideA,
                                                       T* B_,
                                                       I ldb,
                                                       Istride strideB,
                                                       T* C_,
                                                       I ldc,
                                                       Istride strideC,
                                                       I devinfo_array[],
                                                       I batchCount)
{
    /* 
    ---------------
    check arguments
    ---------------
    */
    if(handle == nullptr)
    {
        return (rocblas_status_invalid_handle);
    };

    // no work
    if((nb == 0) || (nblocks == 0) || (batchCount == 0))
    {
        return (rocblas_status_success;)
    };

    if((A_ == nullptr) || (B_ == nullptr) || (C_ == nullptr))
    {
        return (rocblas_status_invalid_pointer);
    };
    {
        bool const isok = (nb >= 1) && (nblocks >= 1) && (batchCount >= 1) && (strideA >= 1)
            && (strideB >= 1) && (strideC >= 1) && (lda >= nb) && (ldb >= nb) && (ldc >= nb);
        if(!isok)
        {
            return (rocblas_status_invalid_size);
        };
    };

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    geblttrf_npvt_strided_batched_template<T, I, Istride>(stream, nb, nblocks, batchCount, A_, lda,
                                                          strideA, B_, ldb, strideB, C_, ldc,
                                                          strideC, devinfo_array);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};

extern "C" {

rocblas_status rocsolver_dgeblttrf_strided_batched(rocblas_handle handle,
                                                   rocblas_int nb,
                                                   rocblas_int nblocks,
                                                   double* A_,
                                                   rocblas_int lda,
                                                   rocblas_stride strideA,
                                                   double* B_,
                                                   rocblas_int ldb,
                                                   rocblas_stride strideB,
                                                   double* C_,
                                                   rocblas_int ldc,
                                                   rocblas_stride strideC,
                                                   rocblas_int devinfo_array[],
                                                   rocblas_int batchCount)
{
    return (rocsolver_geblttrf_strided_batched_impl<double, rocblas_int, rocblas_stride>(
        handle, nb, nblocks, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC, devinfo_array,
        batchCount));
};

rocblas_status rocsolver_sgeblttrf_strided_batched(rocblas_handle handle,
                                                   rocblas_int nb,
                                                   rocblas_int nblocks,
                                                   float* A_,
                                                   rocblas_int lda,
                                                   rocblas_stride strideA,
                                                   float* B_,
                                                   rocblas_int ldb,
                                                   rocblas_stride strideB,
                                                   float* C_,
                                                   rocblas_int ldc,
                                                   rocblas_stride strideC,
                                                   rocblas_int devinfo_array[],
                                                   rocblas_int batchCount)
{
    return (rocsolver_geblttrf_strided_batched_impl<float, rocblas_int, rocblas_stride>(
        handle, nb, nblocks, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC, devinfo_array,
        batchCount));
};

rocblas_status rocsolver_zgeblttrf_strided_batched(rocblas_handle handle,
                                                   rocblas_int nb,
                                                   rocblas_int nblocks,
                                                   rocblas_double_complex* A_,
                                                   rocblas_int lda,
                                                   rocblas_stride strideA,
                                                   rocblas_double_complex* B_,
                                                   rocblas_int ldb,
                                                   rocblas_stride strideB,
                                                   rocblas_double_complex* C_,
                                                   rocblas_int ldc,
                                                   rocblas_stride strideC,
                                                   rocblas_int devinfo_array[],
                                                   rocblas_int batchCount)
{
    return (
        rocsolver_geblttrf_strided_batched_impl<rocblas_double_complex, rocblas_int, rocblas_stride>(
            handle, nb, nblocks, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,
            devinfo_array, batchCount));
};

rocblas_status rocsolver_cgeblttrf_strided_batched(rocblas_handle handle,
                                                   rocblas_int nb,
                                                   rocblas_int nblocks,
                                                   rocblas_float_complex* A_,
                                                   rocblas_int lda,
                                                   rocblas_stride strideA,
                                                   rocblas_float_complex* B_,
                                                   rocblas_int ldb,
                                                   rocblas_stride strideB,
                                                   rocblas_float_complex* C_,
                                                   rocblas_int ldc,
                                                   rocblas_stride strideC,
                                                   rocblas_int devinfo_array[],
                                                   rocblas_int batchCount)
{
    return (
        rocsolver_geblttrf_strided_batched_impl<rocblas_float_complex, rocblas_int, rocblas_stride>(
            handle, nb, nblocks, A_, lda, strideA, B_, ldb, strideB, C_, ldc, strideC,
            devinfo_array, batchCount));
};
}
