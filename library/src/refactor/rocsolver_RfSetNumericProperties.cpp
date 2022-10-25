#include "rocsolver_refactor.h"

rocsolverStatus_t rocsolverRfSetNumericProperties(
		  rocsolverRfHandle_t handle,
		  double effective_zero,
		  double boost_val)
{
    if (handle == nullptr) {
      return( ROCSOLVER_STATUS_NOT_INITIALIZED );
      };

    handle->effective_zero = effective_zero;
    handle->boost_val = boost_val;

    return( ROCSOLVER_STATUS_SUCCESS );
}
