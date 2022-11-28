
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
#include "rocsolver_geblttrs_npvt_batched.hpp"

template <typename T, typename I>
rocblas_status rocsolver_geblttrs_batched_impl(rocblas_handle handle,
                                               I nb,
                                               I nblocks,
                                               I nrhs,
                                               T* A_array[],
                                               I lda,
                                               T* B_array[],
                                               I ldb,
                                               T* C_array[],
                                               I ldc,
                                               T* brhs_,
                                               I ldbrhs,
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
    if((nb == 0) || (nblocks == 0) || (batchCount == 0) || (nrhs == 0))
    {
        return (rocblas_status_success);
    };

    if((A_array == nullptr) || (B_array == nullptr) || (C_array == nullptr) || (brhs_ == nullptr))
    {
        return (rocblas_status_invalid_pointer);
    };

    {
        bool const isok = (nb >= 1) && (nblocks >= 1) && (batchCount >= 1) && (lda >= nb)
            && (ldb >= nb) && (ldc >= nb) && (ldbrhs >= (nb * nblocks));
        if(!isok)
        {
            return (rocblas_status_invalid_size);
        };
    };

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    I host_info = 0;
    geblttrs_npvt_batched_template<T, I>(stream, nb, nblocks, nrhs, batchCount, A_array, lda,
                                         B_array, ldb, C_array, ldc, brhs_, ldbrhs, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};

extern "C" {

rocblas_status rocsolver_dgeblttrs_batched(rocblas_handle handle,
                                           rocblas_int nb,
                                           rocblas_int nblocks,
                                           rocblas_int nrhs,
                                           double* A_array[],
                                           rocblas_int lda,
                                           double* B_array[],
                                           rocblas_int ldb,
                                           double* C_array[],
                                           rocblas_int ldc,
                                           double* brhs_,
                                           rocblas_int ldbrhs,
                                           rocblas_int batchCount)
{
    return (rocsolver_geblttrs_batched_impl<double, rocblas_int>(handle, nb, nblocks, nrhs, A_array,
                                                                 lda, B_array, ldb, C_array, ldc,
                                                                 brhs_, ldbrhs, batchCount));
};
}
