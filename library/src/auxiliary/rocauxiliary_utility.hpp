/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/
#pragma once

#if(0)
#include "lapack_device_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#endif

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"

#include <cmath>
#include <complex>

ROCSOLVER_BEGIN_NAMESPACE

static int get_max_threads()
{
    return (1024);
}

static int get_num_cu(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMultiprocessorCount;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

static int get_lds_size()
{
    int const default_lds_size = 64 * 1024;

    int lds_size = 0;
    int deviceId = 0;
    auto const istat_device = hipGetDevice(&deviceId);
    if(istat_device != hipSuccess)
    {
        return (default_lds_size);
    };
    auto const attr = hipDeviceAttributeMaxSharedMemoryPerBlock;
    auto istat_attr = hipDeviceGetAttribute(&lds_size, attr, deviceId);
    if(istat_attr != hipSuccess)
    {
        return (default_lds_size);
    };

    return (lds_size);
}
ROCSOLVER_END_NAMESPACE
