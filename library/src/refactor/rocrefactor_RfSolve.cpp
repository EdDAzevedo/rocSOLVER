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
                   (Temp != nullptr) && ( ldt >= n) &&
                   (XF != nullptr) && (ldxf >= n);
 if (!isok_arguments) {
   return( ROCSOLVER_STATUS_INVALID_VALUE );
   };


























  

       
} 
