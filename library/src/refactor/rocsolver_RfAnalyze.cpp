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
