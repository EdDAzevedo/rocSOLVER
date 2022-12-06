
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
#ifndef CAL_INDEX_H
#define CAL_INDEX_H

#include <assert.h>

#define is_ijk(nrow, ncol, ld, stride, batchCount) (((ld) >= (nrow)) && ((stride) >= (ld) * (ncol)))

#define is_ikj(nrow, ncol, ld, stride, batchCount) \
    (((ld) >= (stride) * (batchCount)) && ((stride) >= (nrow)))

#define is_kij(nrow, ncol, ld, stride, batchCount) (((stride) == 1) && ((ld) >= (nrow)))

#define CAL_INDEX(nrow, ncol, ld, stride, batchCount, ci, cj, ck)            \
    {                                                                        \
        bool const is_ijk0 = is_ijk(nrow, ncol, ld, stride, batchCount);     \
        bool const is_ikj0 = is_ikj(nrow, ncol, ld, stride, batchCount);     \
        bool const is_kij0 = is_kij(nrow, ncol, ld, stride, batchCount);     \
        int ncase = 0;                                                       \
        if(is_ijk0)                                                          \
        {                                                                    \
            ncase++;                                                         \
        };                                                                   \
        if(is_ikj0)                                                          \
        {                                                                    \
            ncase++;                                                         \
        };                                                                   \
        if(is_kij0)                                                          \
        {                                                                    \
            ncase++;                                                         \
        };                                                                   \
        bool const is_valid = (ncase == 1);                                  \
        assert(is_valid);                                                    \
        if(is_ijk0)                                                          \
        {                                                                    \
            /*  A(i,j,k)  A_[ i + j*ld + k * stride ] */                     \
            ci = 1;                                                          \
            cj = ld;                                                         \
            ck = stride;                                                     \
        };                                                                   \
        if(is_ikj0)                                                          \
        {                                                                    \
            /*  A(i,j,k)  A_[ i + k * stride + j * ld ] */                   \
            ci = 1;                                                          \
            cj = ld;                                                         \
            ck = stride;                                                     \
        };                                                                   \
        if(is_kij0)                                                          \
        {                                                                    \
            /* A(i,j,k)  A_[ k + i * batchCount + j * (batchCount * ld) ] */ \
            ci = batchCount;                                                 \
            ck = 1;                                                          \
            cj = batchCount;                                                 \
            cj *= ld;                                                        \
        };                                                                   \
    }

#endif
