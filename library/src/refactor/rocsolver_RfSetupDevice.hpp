
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

template <typename Iint, typename Ilong, typename T>
rocsolverStatus_t rocsolver_RfSetup_checkargs(Iint n,

                                              Ilong nnzA,
                                              Ilong* csrRowPtrA,
                                              Iint* csrColIndA,
                                              T* csrValA,

                                              Ilong nnzL,
                                              Ilong* csrRowPtrL,
                                              Iint* csrColIndL,
                                              T* csrValL,

                                              Ilong nnzU,
                                              Ilong* csrRowPtrU,
                                              Iint* csrColIndU,
                                              T* csrValU,

                                              int* P,
                                              int* Q,
                                              rocsolverRfHandle_t handle)
{
    if(csrValA == 0)
    {
        return (ROCSOLVER_STATUS_INVALID_VALUE);
    };

      
    Iint constexpr batch_count = 1;

    return (rocsolver_RfBatchSetup_checkargs(
                                        batch_count, n, 
                                        nnzA, csrRowPtrA, csrColIndA, &csrValA, 
                                        nnzL, csrRowPtrL, csrColIndL, csrValL, 
                                        nnzU, csrRowPtrU, csrColIndU, csrValU, 
                                        P, Q, handle));
};

template <bool MAKE_COPY, typename Iint, typename Ilong, typename T>
rocsolverStatus_t rocsolverRfSetupDevice_impl(/* Input (in the device memory) */
                                              Iint n,

                                              Ilong nnzA,
                                              Ilong* csrRowPtrA_in,
                                              Iint* csrColIndA_in,
                                              T* csrValA_in,

                                              Ilong nnzL,
                                              Ilong* csrRowPtrL_in,
                                              Iint* csrColIndL_in,
                                              T* csrValL_in,

                                              Ilong nnzU,
                                              Ilong* csrRowPtrU_in,
                                              Iint* csrColIndU_in,
                                              T* csrValU_in,

                                              Iint* P_in,
                                              Iint* Q_in,

                                              /* Output */
                                              rocsolverRfHandle_t handle)
{

    if(csrValA_in == nullptr)
    {
        return (ROCSOLVER_STATUS_INVALID_VALUE);
    };

    Iint constexpr batch_count = 1;

    return (rocsolverRfBatchSetupDevice_imp<MAKE_COPY,Iint,Ilong,T>(
                                            batch_count, n, 
                                            nnzA, csrRowPtrA_in, csrColIndA_in, &csrValA_in, 
                                            nnzL, csrRowPtrL_in, csrColIndL_in, csrValL_in, 
                                            nnzU, csrRowPtrU_in, csrColIndU_in, csrValU_in, 
                                            P_in, Q_in, handle));
}
