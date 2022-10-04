/*
 -------------------------------------------------------------
 This routine performs the forward and backward solve with the 
 upper and lower triangular factors computed from 
 hipsolverRfRefactor()
 -------------------------------------------------------------
*/
rocsolverStatus_t rocrefactor_RfSolve(
              /* Input (in the device memory) */
              rocsolverRfHandle_t handle,
              int *P,
              int *Q,
              int nrhs,
              double *Temp, /* dense matrix of size (ldt * nrhs), ldt >= n */
              int ldt,

              /* Input/Output (in the device memory) */

              /* 
               -----------------------------------------
               dense matrix that contains right-hand side F
               and solutions X of size (ldxf * nrhs)
               -----------------------------------------
              */
              double *XF,  
                
              /* Input */
              int ldxf
              )
{
  if (handle == nullptr) {
    return( ROCSOLVER_STATUS_NOT_INITIALIZED );
    };

 int const n = handle->n;
 if (n <= 0) {
    return( ROCSOLVER_STATUS_SUCCESS );
    };

 bool const isok_arguments = 
                   (XF != nullptr) && (ldxf >= n);
 if (!isok_arguments) {
   return( ROCSOLVER_STATUS_INVALID_VALUE );
   };


 int * const P_new2old = P;
 int * const Q_new2old = Q;

 assert( P_new2old == handle->P_new2old );
 assert( Q_new2old == handle->Q_new2old );

 for(int irhs=0; irhs < nrhs; irhs++) {
   double * const Rs = nullptr;
   double * const brhs = &(XF[ ldxf*irhs ]);
   int * const LUp = handle->csrrowPtrLU;
   int * const LUi = handle->csrColIndLU;
   double * const LUx = handle->csrValLU;

   int isok = rf_pqrlusolve( 
                           handle->hipsparse_handle,
                           n,
                           P_new2old,
                           Q_new2old,
                           Rs,
                           Lup,
                           Lui,
                           Lux,
                           brhs
                           );
   if (!isok) {
      return( ROCSOLVER_STATUS_INTERNAL_ERROR );
   };


return( ROCSOLVER_STATUS_SUCCESS );
       
} 
