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
#ifndef HIPSPARSE_CXX_HPP
#define HIPSPARSE_CXX_HPP

#include <cassert>
#include <iostream>
#include <stdexcept>

#include "hip_cxx.hpp"
#include "hipsparse/hipsparse.h"
/*
 --------------------------------------
 wrap hipsparse handles and object
 to allow C++ compiler to automatically
 setup or deallocate resources
 --------------------------------------
*/

struct hipsparseHandle_cxx_t
{
    hipsparseHandle_t data = nullptr;

    hipsparseHandle_cxx_t()
    {
        hipsparseStatus_t istat_create = hipsparseCreate(&data);
        bool const isok_create = (istat_create == HIPSPARSE_STATUS_SUCCESS) && (data != nullptr);
        if(!isok_create)
        {
            throw std::runtime_error("hipsparseHandle_cxx_t create");
        };
    };

    ~hipsparseHandle_cxx_t()
    {
        hipsparseStatus_t istat_destroy = hipsparseDestroy(data);
        data = nullptr;
        bool const isok_destroy = (istat_destroy == HIPSPARSE_STATUS_SUCCESS);

        // note: destructor should not throw
        assert(isok_destroy);
    };
};


struct hipsparseMatDescr_cxx_t {
  hipsparseMatDescr_t data = nullptr;

  hipsparseMatDescr_cxx_t( hipsparseMatrixType_t type = HIPSPARSE_MATRIX_TYPE_GENERAL ) {
    hipsparseStatus_t const istat_Create = hipsparseCreateMatDescr( &data );
    bool const isok_Create = (istat_Create == HIPSPARSE_STATUS_SUCCESS);
    if (!isok_Create) { 
        throw std::runtime_error("hipsparseMatDescr_cxx_t Create");
        };

    hipsparseStatus_t  const istat_SetMatType = hipsparseSetMatType( data, type );
    bool const isok_SetMatType = (istat_SetMatType == HIPSPARSE_STATUS_SUCCESS );
    if (!isok_SetMatType) { 
        throw std::runtime_error("hipsparseMatDescr_cxx_t SetMatType");
        };
    
    };

  ~hipsparseMatDescr_cxx_t() {
    hipsparseStatus_t istat = hipsparseDestroyMatDescr( data );
    bool const isok = (istat == HIPSPARSE_STATUS_SUCCESS );
  
    // note: destructor cannot throw
    assert( isok );
    };
};



#endif
