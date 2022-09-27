#ifndef ROCREFACTOR_RFRESETVALUES_HPP
#define ROCREFACTOR_RFRESETVALUES_HPP

#include "rocsolverRf.h"

#include "rocrefactor_aXpbY.hpp"

template< typename Iint, typename Ilong, typename T>
rocsolverStatus_t 
rocsolverRfResetValues(
                     Iint n,
                     Iint nnzA,
                     Iint* csrRowPtrA,
                     Iint* csrColIndA,
                     T* csrValA,
                     Iint* P,
                     Iint* Q,
          
                     rocsolverRfHandle_t handle
                     )
{

  /*
   ------------
   Quick return
   ------------
   */
  if ((n <= 0) || (nnzA <= 0)) {
    return( ROCSOLVER_STATUS_SUCCESS );
    };

  bool const isok =  (csrRowPtrA != NULL) &&
                     (csrColIndA != NULL) &&
                     (csrValA != NULL) &&
                     (handle != NULL);
                     
  if (!isok) {
        return( ROCSOLVER_STATUS_INVALID_VALUE );
        };

  hipStream_t stream;
  HIPSPARSE_CHECK( hipsparseGetStream( handle->hipsparse_handle, &stream ) );

  if ((P == NULL) && (Q == NULL)) {

  int const nrow = n;
  int const ncol = n;
  double const alpha = 1;
  double const beta  = 0;
  int const * const Xp = csrRowPtrA;
  int const * const Xi = csrColIndA;
  double const * const Xx = csrValA;

  int const * const Yp = handle->csrRowPtrLU;
  int const * const Yi = handle->csrColIndLU;
  double const * const Yx = handle->csrValLU;
  rocrefactor_aXpbY<Iint,Ilong,T>(
                     stream,

                     nrow,
                     ncol,
                     alpha,
                     Xp,
                     Xi,
                     Xx,
                     beta,
                     Yp,
                     Yi,
                     Yx
                     );
  }
 else {

  int const * const P_new2old = P;
  int const * const Q_new2old = Q;
  int const * const Q_old2new = handle->Q_old2new;

  int const * const * Ap = csrRowPtrA;
  int const * const * Ai = csrColIndA;
  double const * const Ax = csrValA;

  int const * const LUp = handle->csrRowPtrLU;
  int const * const LUi = handle->csrColIndLU;
  double const * const LUx = handle->csrValLU;
   
  rocrefactor_add_PAQ<Iint,Ilong,T>(
                     stream,
                     nrow,
                     ncol,
                     P_new2old,
                     Q_old2new,
                     Ap,
                     Ai,
                     Ax,
                     LUp,
                     LUi,
                     LUx
                     );


  };

 return( ROCSOLVER_STATUS_SUCCESS );
}




#endif
