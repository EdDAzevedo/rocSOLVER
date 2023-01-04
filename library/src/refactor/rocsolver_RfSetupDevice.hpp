
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
#pragma once
#include "rf_common.hpp"

#include "rocsolver_RfBatchSetupDevice.hpp"


template <typename T>
rocsolverStatus_t rocsolver_RfSetup_checkargs(int n,
                                              int nnzA,
                                              int* csrRowPtrA,
                                              int* csrColIndA,
                                              T* csrValA,
                                              int nnzL,
                                              int* csrRowPtrL,
                                              int* csrColIndL,
                                              T* csrValL,
                                              int nnzU,
                                              int* csrRowPtrU,
                                              int* csrColIndU,
                                              T* csrValU,
                                              int* P,
                                              int* Q,
                                              rocsolverRfHandle_t handle)
{

   int const batch_count = 1;
   if (csrValA == 0) {
      return( ROCSOLVER_STATUS_INVALID_VALUE );
      };
   return( rocsolver_RfSetup_checkargs(
                              batch_count
                              n,
                              nnzA, csrRowPtrA, csrColIndA, &csrValA,
                              nnzL, csrRowPtrL, csrColIndL, csrValL,
                              nnzU, csrRowPtrU, csrColIndU, csrValU,
                              P, Q,
                              handle ) );
};

template <bool MAKE_COPY>
rocsolverStatus_t rocsolverRfSetupDevice_impl(/* Input (in the device memory) */
                                              int n,
                                              int nnzA,
                                              int* csrRowPtrA_in,
                                              int* csrColIndA_in,
                                              double* csrValA_in,
                                              int nnzL,
                                              int* csrRowPtrL_in,
                                              int* csrColIndL_in,
                                              double* csrValL_in,
                                              int nnzU,
                                              int* csrRowPtrU_in,
                                              int* csrColIndU_in,
                                              double* csrValU_in,
                                              int* P_in,
                                              int* Q_in,

                                              /* Output */
                                              rocsolverRfHandle_t handle)
{
    int const batch_count = 1;

    if (csrValA_in == 0) {
       return( ROCSOLVER_STATUS_INVALID_VALUE );
       };

    return( rocsolverRfBatchSetupDevice_imp(
                    batch_count,
                    n,
                    nnzA, csrRowPtrA_in, csrColIndA_in, &csrValA_in,
                    nnzL, csrRowPtrL_in, csrColIndL_in, csrValL_in,
                    nnzU, csrRowPtrU_in, csrColIndU_in, csrValU_in,
                    P_in, Q_in,
                    handle ) );

}
