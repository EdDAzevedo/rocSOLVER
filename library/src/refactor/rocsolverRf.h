#ifndef ROCSOLVERRF_H
#define ROCSOLVERRF_H

#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"
#include "rocsolver_refactor.h"

#ifdef __cplusplus
extern "C" {
#endif


rocsolverStatus_t
rocsolverRfResetValues(
           int n,
           int nnzA,
           int* csrRowPtrA,
           int* csrColIndA,
           double* csrValA,
           int* P,
           int* Q,

           rocsolverRfHandle_t handle
           );

#ifdef __cplusplus
};
#endif


#endif
