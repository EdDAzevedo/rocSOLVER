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

#include "rf_common.hpp"

/*
-----------------------------------------------------------------------
This routine shuts down the rocSolverRF library.  It releases acquired
resources and must be called after all the rocsolverRF library routines.
-----------------------------------------------------------------------
*/

extern "C" {

rocsolverStatus_t rocsolverRfDestroy(rocsolverRfHandle_t handle)
{
    if(handle == nullptr)
    {
        return (ROCSOLVER_STATUS_NOT_INITIALIZED);
    };

    rocsolverStatus_t istat_return = ROCSOLVER_STATUS_SUCCESS;

    try
    {
        delete handle;
    }
    catch(const std::bad_alloc& e)
    {
        istat_return = ROCSOLVER_STATUS_ALLOC_FAILED;
    }
    catch(const std::runtime_error& e)
    {
        istat_return = ROCSOLVER_STATUS_EXECUTION_ERROR;
    }
    catch(...)
    {
        istat_return = ROCSOLVER_STATUS_INTERNAL_ERROR;
    };

    return (istat_return);
};
};
