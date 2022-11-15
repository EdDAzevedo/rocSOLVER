
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

extern "C" {

rocblas_status rocsolverDgbtrfBatched(rocblas_handle handle,
                                      int nb,
                                      int nblocks,
                                      double* A_array[],
                                      int lda,
                                      double* B_array[],
                                      int ldb,
                                      double* C_array[],
                                      int ldc,
                                      int batchCount)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    int host_info = 0;
    gbtrf_npvt_batched_template<double>(stream, nb, nblocks, batchCount, A_array, lda, B_array, ldb,
                                        C_array, ldc, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};

rocblas_status rocsolverSgbtrfBatched(rocblas_handle handle,
                                      int nb,
                                      int nblocks,
                                      float* A_array[],
                                      int lda,
                                      float* B_array[],
                                      int ldb,
                                      float* C_array[],
                                      int ldc,
                                      int batchCount)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    int host_info = 0;
    gbtrf_npvt_batched_template<float>(stream, nb, nblocks, batchCount, A_array, lda, B_array, ldb,
                                       C_array, ldc, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};

rocblas_status rocsolverCgbtrfBatched(rocblas_handle handle,
                                      int nb,
                                      int nblocks,
                                      rocblas_float_complex* A_array[],
                                      int lda,
                                      rocblas_float_complex* B_array[],
                                      int ldb,
                                      rocblas_float_complex* C_array[],
                                      int ldc,
                                      int batchCount)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    int host_info = 0;
    gbtrf_npvt_batched_template<rocblas_float_complex>(stream, nb, nblocks, batchCount, A_array,
                                                       lda, B_array, ldb, C_array, ldc, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};

rocblas_status rocsolverZgbtrfBatched(rocblas_handle handle,
                                      int nb,
                                      int nblocks,
                                      rocblas_double_complex* A_array[],
                                      int lda,
                                      rocblas_double_complex* B_array[],
                                      int ldb,
                                      rocblas_double_complex* C_array[],
                                      int ldc,
                                      int batchCount)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    int host_info = 0;
    gbtrf_npvt_batched_template<rocblas_double_complex>(stream, nb, nblocks, batchCount, A_array,
                                                        lda, B_array, ldb, C_array, ldc, &host_info);

    return ((host_info == 0) ? rocblas_status_success : rocblas_status_internal_error);
};
}
