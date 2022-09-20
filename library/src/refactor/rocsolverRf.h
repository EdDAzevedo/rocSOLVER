#ifndef ROCSOLVERRF_H
#define ROCSOLVERRF_H

#include "rocblas.h"
#include "rocsolver.h"
#include "rocrefactor.h"

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

           rocsolverRFHandle_t handle
           );

#ifdef __cplusplus
};
#endif


#endif
