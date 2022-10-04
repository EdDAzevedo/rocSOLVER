#ifndef RF_LUSOLVE_H
#define RF_LUSOLVE_H

#include "helper_utility.h"
#include <hipsparse/hipsparse.h>
#include "rocsolver_refactor.h"

#ifdef __cplusplus
extern "C" {
#endif

extern
rocsolverStatus_t rf_lusolve( 
                 hipsparseHandle_t handle,
                 int const n, 
                 int const nnz,
                 int * const d_LUp, 
                 int * const d_LUi, 
                 double * const d_LUx, 
                 double * const d_b);


#ifdef __cplusplus
};
#endif

#endif
