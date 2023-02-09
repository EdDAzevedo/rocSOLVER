
/*! \file */
/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSOLVER_RFHANDLE_RELEASE_HPP
#define ROCSOLVER_RFHANDLE_RELEASE_HPP

#include "rf_common.hpp"

/*
-------------------------------
Release array storage in rocsolverRf handle
-------------------------------
*/
static rocsolverStatus_t rocsolverRfHandleRelase_imp(rocsolverRfHandle_t handle,
                          bool const destroy_descr = false)
{
    rocsolverStatus_t const istat_err = ROCSOLVER_STATUS_ALLOC_FAILED;
    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;

    int const batch_count = handle->batch_count;

#define HIP_FREE(var)                                                         \
    {                                                                         \
        if((var) != nullptr)                                                  \
        {                                                                     \
            hipError_t const istat = hipFree(var);                            \
            istat_return = (istat != HIP_SUCCESS) ? istat_err : istat_return; \
            var = nullptr;                                                    \
        };                                                                    \
    };

    HIP_FREE(handle->buffer);

    HIP_FREE(handle->csrRowPtrL);
    HIP_FREE(handle->csrColIndL);
    HIP_FREE(handle->csrValL);

    HIP_FREE(handle->csrRowPtrU);
    HIP_FREE(handle->csrColIndU);
    HIP_FREE(handle->csrValU);

    HIP_FREE(handle->csrRowPtrA);
    HIP_FREE(handle->csrColIndA);

    HIP_FREE(handle->P_new2old);
    HIP_FREE(handle->Q_new2old);
    HIP_FREE(handle->Q_old2new);

    HIP_FREE(handle->csrRowPtrLU);
    HIP_FREE(handle->csrColIndLU);

    if(handle->csrValLU_array != nullptr)
    {
        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            HIP_FREE(handle->csrValLU_array[ibatch]);
        };
        free(handle->csrValLU_array);
        handle->csrValLU_array = nullptr;
    };

  if (destroy_descr) {

#define HIPSPARSE_DESTROY_MATDESCR(var)                                                    \
    {                                                                                      \
        if((var) != nullptr)                                                               \
        {                                                                                  \
            hipsparseStatus_t const istat = hipsparseDestroyMatDescr(var);                 \
            istat_return = (istat != HIPSPARSE_STATUS_SUCCESS) ? istat_err : istat_return; \
            var = nullptr;                                                                 \
        };                                                                                 \
    };

    HIPSPARSE_DESTROY_MATDESCR(handle->descrL);
    HIPSPARSE_DESTROY_MATDESCR(handle->descrU);
    HIPSPARSE_DESTROY_MATDESCR(handle->descrLU);

#define HIPSPARSE_DESTROY_CSRSV2INFO(var)                                                  \
    {                                                                                      \
        if((var) != nullptr)                                                               \
        {                                                                                  \
            hipsparseStatus_t const istat = hipsparseDestroyCsrsv2Info(var);               \
            istat_return = (istat != HIPSPARSE_STATUS_SUCCESS) ? istat_err : istat_return; \
            var = nullptr;                                                                 \
        };                                                                                 \
    };

    HIPSPARSE_DESTROY_CSRSV2INFO(handle->infoL);
    HIPSPARSE_DESTROY_CSRSV2INFO(handle->infoU);

#define HIPSPARSE_DESTROY_CSRILU02INFO(var)                                                \
    {                                                                                      \
        if((var) != nullptr)                                                               \
        {                                                                                  \
            hipsparseStatus_t const istat = hipsparseDestroyCsrilu02Info(var);             \
            istat_return = (istat != HIPSPARSE_STATUS_SUCCESS) ? istat_err : istat_return; \
            var = nullptr;                                                                 \
        };                                                                                 \
    };

    if(handle->infoLU_array != nullptr)
    {
        for(int ibatch = 0; ibatch < batch_count; ibatch++)
        {
            HIPSPARSE_DESTROY_CSRILU02INFO(handle->infoLU_array[ibatch]);
        };
        free(handle->infoLU_array);
        handle->infoLU_array = nullptr;
    };

   };

    handle->batch_count = 0;
    handle->nnzA = 0;
    handle->nnzL = 0;
    handle->nnzU = 0;
    handle->nnzLU = 0;
    handle->buffer_size = 0;

    return (istat_return);
};

#undef HIP_FREE
#undef HIPSPARSE_DESTROY_MATDESCR
#undef HIPSPARSE_DESTROY_CSRSV2INFO
#undef HIPSPARSE_DESTROY_CSRILU02INFO

#endif
