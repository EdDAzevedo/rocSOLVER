#include "hip_check.h"
#include "hipsparse_check.h"
#include "rocsolver_refactor.h"

extern "C"
rocsolverStatus_t
rocsolverRfCreate( rocsolverRfHandle_t *p_handle )
{
 rocsolverRfHandle_t handle;
 HIP_CHECK( hipMalloc( (void **) &handle, sizeof(*handle) ),
            ROCSOLVER_STATUS_ALLOC_FAILED );
 
 if (handle == nullptr) {
   return( ROCSOLVER_STATUS_ALLOC_FAILED );
   };

 HIPSPARSE_CHECK( hipsparseCreate( &(handle->hipsparse_handle) ),
                  ROCSOLVER_STATUS_NOT_INITIALIZED );

 /*
  --------------------
  setup default values
  --------------------
*/
 handle->fast_mode = ROCSOLVERRF_RESET_VALUES_FAST_MODE_ON;
 handle->matrix_format = ROCSOLVERRF_MATRIX_FORMAT_CSR;
 handle->triangular_solve = ROCSOLVERRF_TRIANGULAR_SOLVE_ALG1;
 handle->numeric_boost = ROCSOLVERRF_NUMERIC_BOOST_NOT_USED;

 handle->descrL = nullptr;
 handle->descrU = nullptr;
 handle->descrLU = nullptr;

 handle->P_new2old = nullptr;
 handle->Q_new2old = nullptr;
 handle->Q_old2new = nullptr;

 handle->n = 0;
 handle->nnz_LU = 0;
 handle->csrRowPtrLU = nullptr;
 handle->csrColIndLU = nullptr;
 handle->csrValLU = nullptr;

 handle->effective_zero = 0;
 handle->boost_val = 0;

 *p_handle = handle;
 return( ROCSOLVER_STATUS_SUCCESS );
}
