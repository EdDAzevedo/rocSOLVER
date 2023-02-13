
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

#include "rocsolver_RfAccessBundledFactors.hpp"

/*
-----------------------------------------------------------------------
This routine allows direct access to the lower L and upper U triangular
factors stored in the rocSolverRF library handle. The factors are
compressed into a single matrix M = (L-I) + U, where the unitary
diagonal of L is not stored.  It is assumed that a prior call to the
rocsolverRfRefactor() was done in order to generate these triangular
factors.

The Bundled factor matrix M = (L - I) + U
-----------------------------------------------------------------------
*/
extern "C" {

rocsolverStatus_t rocsolverRfAccessBundledFactors(/* Input */
                                                  rocsolverRfHandle_t handle,
                                                  /* Output (in the host memory ) */
                                                  int* nnzM,
                                                  /* Output (in the device memory) */
                                                  int** Mp,
                                                  int** Mi,
                                                  double** Mx)
{
    return (rocsolverRfAccessBundledFactors_impl(handle, nnzM, Mp, Mi, Mx));
};
};
