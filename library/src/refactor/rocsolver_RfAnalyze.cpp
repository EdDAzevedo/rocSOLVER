
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
#include "hip_check.h"
#include "hipsparse_check.h"

#include "rocsolver_refactor.h"

extern "C" {

rocsolverStatus_t rocsolverRfAnalyze( rocsolverRfHandle_t handle) 
{
 if (handle == nullptr) {
    return( ROCSOLVER_STATUS_NOT_INITIALIZED );
    };

 if (handle->hipsparse_handle == nullptr) {
    return( ROCSOLVER_STATUS_NOT_INITIALIZED );
    };

  int lnnz = handle->nnz_LU;
  int n = handle->n;

  hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;
  hipsparseOperation_t transL =  HIPSPARSE_OPERATION_NON_TRANSPOSE;
  hipsparseOperation_t transU =  HIPSPARSE_OPERATION_NON_TRANSPOSE;


  csrsv2Info_t infoL;
  HIPSPARSE_CHECK(hipsparseCreateCsrsv2Info(&infoL),
                 ROCSOLVER_STATUS_INTERNAL_ERROR);

  csrsv2Info_t infoU;
  HIPSPARSE_CHECK( hipsparseCreateCsrsv2Info(&infoU),
         ROCSOLVER_STATUS_INTERNAL_ERROR);

  size_t bufferSize = 1;
  int stmp = 0;


  stmp = 0;
  HIPSPARSE_CHECK( hipsparseDcsrsv2_bufferSize(
       handle->hipsparse_handle, 
       transL, 
       n, 
       lnnz, 
       handle->descrL, 
       handle->csrValLU, 
       handle->csrRowPtrLU,  
       handle->csrColIndLU,  
       infoL, 
       &stmp),
       ROCSOLVER_STATUS_INTERNAL_ERROR);
  if (stmp > bufferSize) {
    bufferSize = stmp;
  };

  stmp = 0;
  HIPSPARSE_CHECK( hipsparseDcsrsv2_bufferSize(
      handle->hipsparse_handle, 
      transU, 
      n, 
      lnnz, 
      handle->descrU, 
      handle->csrValLU, 
      handle->csrRowPtrLU,  
      handle->csrColIndLU, 
      infoU, 
      &stmp),
      ROCSOLVER_STATUS_INTERNAL_ERROR);

  if (stmp > bufferSize) {
    bufferSize = stmp;
  };


  void *buffer = NULL;
  HIP_CHECK( hipMalloc(&buffer, bufferSize),
            ROCSOLVER_STATUS_ALLOC_FAILED);

  HIPSPARSE_CHECK( hipsparseDcsrsv2_analysis(
	handle->hipsparse_handle,
        transL, 
	n, 
	lnnz, 
	handle->descrL, 
	handle->csrValLU, 
	handle->csrRowPtrLU,
	handle->csrColIndLU, 
	infoL,
	policy,
	buffer),
	ROCSOLVER_STATUS_INTERNAL_ERROR );





  HIPSPARSE_CHECK( hipsparseDcsrsv2_analysis(
      handle->hipsparse_handle, 
      transU, 
      n, 
      lnnz, 
      handle->descrU, 
      handle->csrValLU, 
      handle->csrRowPtrLU,
      handle->csrColIndLU,  
      infoU, 
      policy, 
      buffer),
      ROCSOLVER_STATUS_INTERNAL_ERROR );




 /*
  -----------------------------------------
  clean up and deallocate temporary storage
  -----------------------------------------
  */


  HIPSPARSE_CHECK(hipsparseDestroyCsrsv2Info(infoU),
         ROCSOLVER_STATUS_INTERNAL_ERROR);

  HIPSPARSE_CHECK(hipsparseDestroyCsrsv2Info(infoL),
         ROCSOLVER_STATUS_INTERNAL_ERROR);


  HIP_CHECK( hipFree( (void *) buffer),
               ROCSOLVER_STATUS_ALLOC_FAILED );

  return( ROCSOLVER_STATUS_SUCCESS );

};


}
