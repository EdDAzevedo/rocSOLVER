
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
#ifndef RF_ADDLU_HPP
#define RF_ADDLU_HPP

#include "rf_addLU.hpp"


template <typename Iint, typename Ilong, typename T>
rocsolverStatus_t rf_addLU(rocsolverRfHandle_t handle,
                           Iint const nrow,
                           Iint const ncol,

                           Ilong const* const Lp,
                           Iint const* const Li,
                           T const* const Lx,

                           Ilong const* const Up,
                           Iint const* const Ui,
                           T const* const Ux,

                           Ilong* const LUp,
                           Iint* const LUi,
                           T* const LUx

)
{
    //  ----------------
    //  form (L - I) + U
    //  assume storage for LUp, LUi, LUx has been allocated
    // ---------------------------------------------------

    bool const use_sumLU = true;

    if (use_sumLU) {
       return( rf_sumLU( handle,
                         nrow, ncol,
                         Lp, Li, Lx,
                         Up, Ui, Ux,
                         LUp, LUi, LUx ) );
       };

    {
       return( rf_geamLU( handle,
                          nrow, ncol,
                          Lp, Li, Lx,
                          Up, Ui, Ux,
                          LUp, LUi, LUx ) );
       };
                      
}

#endif
