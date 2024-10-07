/* **************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocsolver_handle.hpp"
#include "rocblas.hpp"

#include <memory>

ROCSOLVER_BEGIN_NAMESPACE

rocblas_status rocsolver_enable_hybrid_mode_impl(rocblas_handle handle, const bool enabled)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    std::shared_ptr<void> handle_ptr;
    ROCBLAS_CHECK(rocblas_internal_get_data_ptr(handle, handle_ptr));
    if(handle_ptr.get() == nullptr)
    {
        handle_ptr = std::make_shared<rocsolver_handle_data_>();
        ROCBLAS_CHECK(rocblas_internal_set_data_ptr(handle, handle_ptr));
    }

    rocsolver_handle_data handle_data = (rocsolver_handle_data)handle_ptr.get();
    handle_data->hybrid_mode_enabled = enabled;
    return rocblas_status_success;
}

bool rocsolver_is_hybrid_mode_enabled_impl(rocblas_handle handle)
{
    if(!handle)
        return false;

    std::shared_ptr<void> handle_ptr;
    ROCBLAS_CHECK(rocblas_internal_get_data_ptr(handle, handle_ptr));

    rocsolver_handle_data handle_data = (rocsolver_handle_data)handle_ptr.get();
    return handle_data != nullptr && handle_data->hybrid_mode_enabled;
}

ROCSOLVER_END_NAMESPACE

extern "C" {

rocblas_status rocsolver_enable_hybrid_mode(rocblas_handle handle, const bool enabled)
try
{
    return rocsolver::rocsolver_enable_hybrid_mode_impl(handle, enabled);
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}

bool rocsolver_is_hybrid_mode_enabled(rocblas_handle handle)
try
{
    return rocsolver::rocsolver_is_hybrid_mode_enabled_impl(handle);
}
catch(...)
{
    return false;
}
}
