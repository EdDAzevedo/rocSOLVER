
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
#include "stdio.h"
#include "stdlib.h"

#include "hip_check.h"
#include "hipsparse_check.h"
#include "rocsolver_refactor.h"

/*
------------------------------------------------------------------------
This routine initializes the rocSolverRF library.  It allocates required
resources and must be called prior to any other rocSolverRF library
routine.
------------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfCreate(rocsolverRfHandle_t* p_handle)
{
    if(p_handle == nullptr)
    {
        return (ROCSOLVER_STATUS_INVALID_VALUE);
    };

    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;
    try
    {
        rocsolverRfHandle_t handle = new rocsolverRfHandle();

        // ----------
        // set stream
        // ----------
        hipsparseStatus_t istat_stream
            = hipsparseSetStream(handle->hipsparse_handle.data(), handle->streamId.data());
        bool const isok_stream = (istat_stream == HIPSPARSE_STATUS_SUCCESS);
        if(!isok_stream)
        {
            throw std::runtime_error(__FILE__);
        };

        *p_handle = handle;
    }
    catch(...)
    {
        istat_return = ROCSOLVER_STATUS_INTERNAL_ERROR;
    };

    return (istat_return);
};
};
