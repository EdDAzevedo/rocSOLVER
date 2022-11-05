
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
#ifndef RF_PQRLUSOLVE_H
#define RF_PQRLUSOLVE_H



#include <stdlib.h>
#include <string.h>

#include "rocsolver/rocsolver.h"

#include "rf_lusolve.h"
#include "rf_pvec.h"
#include "rf_ipvec.h"
#include "rf_applyRs.h"
#include "rf_mirror_pointer.h"


rocsolverStatus_t rf_pqrlusolve( 
                  hipsparseHandle_t handle,
                  int const n,
                  int  *  const P_new2old, 
                  int  *  const Q_new2old, 
                  double  *  const Rs, 
                  int  *  const LUp,
                  int  *  const LUi,
                  double *  const LUx,
                  double *   const brhs );
#endif
