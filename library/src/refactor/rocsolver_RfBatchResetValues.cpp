
/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "rocsolver_RfBatchResetValues.hpp"
#include "rocsolver_refactor.h"

/*
-----------------------------------------------------------------------
This routine updates the internal data structures with the values of the
new coefficient matrix.  It is assumed that the arrays csrRowPtrA[],
csrColIndA[], P[] and Q[] have not changed since the last call to
the rocsolverRfSetupHost() or rocsolverRfSetupDevice() routine. This
assumption reflects the fact that the sparsity pattern of coefficient
matrices as well as reordering to minimize fill-in and pivoting remain
the same in the set of linear systems

    A_i x_i = f_i

This routine may be called multiple times, once for each of the linear
systems:

    A_i x_i = f_i

-----------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfBatchResetValues(
    /* Input (in host memory) */
    int batch_count,
    int n,
    int nnzA,

    /* Input (in the device memory) */
    int* csrRowPtrA,
    int* csrColIndA,
    double* csrValA_array[],
    int* P,
    int* Q,
    /* Output */
    rocsolverRfHandle_t handle)
{
    return (rocsolver_RfBatchResetValues_template(batch_count, n, nnzA, csrRowPtrA, csrColIndA,
                                                  csrValA_array, P, Q, handle));
};
}
