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

#pragma once
#ifndef RF_SHELLSORT_HPP
#define RF_SHELLSORT_HPP

#include "rf_common.hpp"

template <typename Iint, typename T>
static __device__ void rf_shellsort(Iint* iarr, T* darr, Iint num)
{
    // ----------------------------------------------------
    // device (serial) code to perform shell sort by a single thread
    // key is  iarr[]
    // data is darr[]
    // ----------------------------------------------------

    for(Iint i = num / 2; i > 0; i = i / 2)
    {
        for(Iint j = i; j < num; j++)
        {
            for(Iint k = j - i; k >= 0; k = k - i)
            {
                if(iarr[k + i] >= iarr[k])
                {
                    break;
                }
                else
                {
                    // swap entries
                    Iint itmp = iarr[k];
                    iarr[k] = iarr[k + i];
                    iarr[k + i] = itmp;

                    T dtmp = darr[k];
                    darr[k] = darr[k + i];
                    darr[k + i] = dtmp;
                };
            };
        };
    };
}
#endif
