
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
#include "hip_check.h"
#include "hipsparse_check.h"

#include "rocsolver_refactor.h"

/*
--------------------------------------------------------------------------
The user can query which matrix failed LU refactorization by checking
corresponding value in "position" array. The input parameter "position" is an
integer array of size "batchSize".

The j-th component denotes the refactorization result of matrix "A(j)".
If position(j) is -1, the LU refactorization of matrix "A(j)" is successful.
If position(j) is k >= 0, matrix "A(j)" is not LU factorizable and entry
"U(j,j)" is zero.
--------------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfBatchZeroPivot(rocsolverRfHandle_t handle,
                                            /* host output */
                                            int* position)
{
    if(handle == 0)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    if(handle->hipsparse_handle == 0)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    int const batch_count = handle->batch_count;

    for(int ibatch = 0; ibatch < batch_count; ibatch++)
    {
        int ipos = 0;
        hipsparseStatus_t istat = hipsparseXcsrilu02_zeroPivot(handle->hipsparse_handle,
                                                               handle->infoLU_array[ibatch], &ipos);
        position[ibatch] = (istat == HIPSPARSE_STATUS_ZERO_PIVOT) ? (ipos - 1) : -1;
    };

    int nerrors = 0;
    for(int ibatch = 0; ibatch < batch_count; ibatch++)
    {
        bool const is_ok = (position[ibatch] == -1);
        if(!is_ok)
        {
            nerrors++;
        };
    };

    return ((nerrors == 0) ? ROCSOLVER_STATUS_SUCCESS : ROCSOLVER_STATUS_ZERO_PIVOT);
};
};
