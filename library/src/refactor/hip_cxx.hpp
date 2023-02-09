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
#ifndef HIP_CXX_HPP
#define HIP_CXX_HPP

#include <cassert>
#include <iostream>
#include <stdexcept>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
/*
 --------------------------------------
 wrap hip object
 to allow C++ compiler to automatically
 setup or deallocate resources
 --------------------------------------
*/

struct hipStream_cxx_t
{
    hipStream_t data = nullptr;

    hipStream_cxx_t()
    {
        hipError_t istat = hipStreamCreate(&data);
        bool const isok = (data != nullptr) && (istat == HIP_SUCCESS);
        if(!isok)
        {
            throw std::runtime_error("hipStream_cxx_t");
        };
    };

    ~hipStream_cxx_t()
    {
        hipError_t istat = hipStreamDestroy(data);
        bool const isok = (istat == HIP_SUCCESS);
        data = nullptr;
        // note: destructor should not throw
        assert(isok);
    };
};

#endif
