/************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas.hpp"
#include "roclapack_gesv.hpp"
#include "roclapack_getrf.hpp"
#include "roclapack_getrs.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_logger.hpp"
#include <rocblas/internal/rocblas-complex-types.h>
#include <rocblas/internal/rocblas-types.h>
#include <rocblas/internal/rocblas_bfloat16.h>

#include "roclapack_getrf_mxp.hpp"

ROCSOLVER_BEGIN_NAMESPACE

template <template <typename...> class Lambda, typename... Ts, typename... Args>
rocblas_status rocsolver_ex_datatype_dispatch(Args&&... args)
{
    return Lambda<Ts...>{}(std::forward<Args>(args)...);
}

template <template <typename...> class Lambda, typename... Ts, typename... Args>
rocblas_status rocsolver_ex_datatype_dispatch(rocblas_datatype dt, Args&&... args)
{
    switch(dt)
    {
    case rocblas_datatype_f16_r:
        return rocsolver_ex_datatype_dispatch<Lambda, rocblas_half, Ts..., Args...>(
            std::forward<Args>(args)...);
    case rocblas_datatype_f32_r:
        return rocsolver_ex_datatype_dispatch<Lambda, float, Ts..., Args...>(
            std::forward<Args>(args)...);
    case rocblas_datatype_f64_r:
        return rocsolver_ex_datatype_dispatch<Lambda, double, Ts..., Args...>(
            std::forward<Args>(args)...);
    case rocblas_datatype_f32_c:
        return rocsolver_ex_datatype_dispatch<Lambda, rocblas_float_complex, Ts..., Args...>(
            std::forward<Args>(args)...);
    case rocblas_datatype_f64_c:
        return rocsolver_ex_datatype_dispatch<Lambda, rocblas_double_complex, Ts..., Args...>(
            std::forward<Args>(args)...);
    case rocblas_datatype_bf16_r:
        return rocsolver_ex_datatype_dispatch<Lambda, rocblas_bfloat16, Ts..., Args...>(
            std::forward<Args>(args)...);
    default: return rocblas_status_not_implemented;
    }
}

ROCSOLVER_END_NAMESPACE
