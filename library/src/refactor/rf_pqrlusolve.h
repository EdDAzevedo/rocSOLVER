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

#include "blas.h"

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
