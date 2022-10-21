
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipsparse/hipsparse.h>



#include "rf_pqrlusolve.h"
rocsolverStatus_t rf_pqrlusolve( 
                  hipsparseHandle_t handle,
                  int const n,
                  int  *  const P_new2old, 
                  int  *  const Q_new2old, 
                  double  *  const Rs, 
                  int  *  const LUp, 
                  int  *  const LUi, 
                  double * const LUx, /* LUp,LUi,LUx  are in CSR format */
                  double * const brhs )
{
  /*
    -------------------------------------------------
    Rs \ (P * A * Q) = LU
    solve A * x = b
       P A Q * (inv(Q) x) = P b
       { Rs \ (P A Q) } * (inv(Q) x) = Rs \ (P b)
       
       (LU) xhat = bhat,  xhat = inv(Q) x, or Q xhat = x,
                          bhat = Rs \ (P b)
    -------------------------------------------------
  */

   bool const isok_arg = (LUp != nullptr) &&
                         (LUi != nullptr) &&
                         (Lux != nullptr) &&
                         (brhs != nullptr);
  if (!isok_arg) {
     return( ROCSOLVER_STATUS_INVALID_VALUE );
     };


  bool isok;

  bool const need_P = (P_new2old != NULL);
  bool const need_Q = (Q_new2old != NULL);
  bool const need_Rs = (Rs != NULL);



         /*
	  ---------------
	  setup hipsparse
	  ---------------
	  */

    bool const need_handle = (handle == nullptr);
    if (need_handle) {
	 HIPSPARSE_CHECK( hipsparseCreate( &handle ),
                          ROCSOLVER_STATUS_INTERNAL_ERROR );
         HIPSPARSE_CHECK( hipsparseSetPointerMode( handle, HIPSPARSE_POINTER_MODE_HOST),
                          ROCSOLVER_STATUS_INTERNAL_ERROR );

	  /*
	   --------------------
	   setup stream on GPU
	   --------------------
	   */
          hipStream_t stream;
	  HIP_CHECK( hipStreamCreate( &stream ),
                     ROCSOLVER_STATUS_INTERNAL_ERROR);
	  HIPSPARSE_CHECK( hipsparseSetStream( handle, stream ),
                     ROCSOLVER_STATUS_INTERNAL_ERROR );
          };


  bool const alloc_brhs = !(is_device_pointer(brhs));
  double * d_brhs = (alloc_brhs) ? 
                      (double *) rf_mirror_pointer( brhs, sizeof(double)*n ) :
                      brhs;

  double * d_bhat = NULL;
  {
  size_t nbytes = sizeof(double)*n;
  HIP_CHECK( hipMalloc( (void **) &d_bhat, nbytes),
             ROCSOLVER_STATUS_ALLOC_ERROR );
  assert( d_bhat != NULL );

  int const value = 0xff;
  HIP_CHECK( hipMemset( (void *) d_bhat, value, nbytes ),
             ROCSOLVER_STATUS_INTERNAL_ERROR ); 
  };




  
  if (need_P) {

    bool const alloc_P_new2old = !(is_device_pointer(P_new2old));
    int * d_P_new2old = (alloc_P_new2old) ?
                           (int *) rf_mirror_pointer( P_new2old, sizeof(int) * n ) :
                           P_new2old;
    assert( d_P_new2old != NULL );





     /*
      ------------------------------
      bhat[k] = brhs[ P_new2old[k] ]
      ------------------------------
      */

     rfDpvec(handle, n, d_P_new2old, d_brhs, d_bhat );

     if (alloc_P_new2old) {
        HIP_CHECK( hipFree( (void *) d_P_new2old ),
                   ROCSOLVER_STATUS_ALLOC_ERROR );
        d_P_new2old = NULL;
        };
 
   } 
   else {
     void * const src =  (void *) d_brhs;
     void * const dest = (void *) d_bhat;
     size_t const nbytes = sizeof(double) * n;
  
     HIP_CHECK( hipMemcpy( dest, src, nbytes, hipMemcpyDeviceToDevice ),
                ROCSOLVER_STATUS_INTERNAL_ERROR);

   };

  




  if (need_Rs) {

    bool const alloc_Rs = !(is_device_pointer(Rs));
    double * d_Rs = (alloc_Rs) ?
                     (double *) rf_mirror_pointer( Rs, sizeof(double)*n ) :
                     Rs;


    rfDapplyRs( handle, n, d_Rs, d_bhat );

    if (alloc_Rs) {
        HIP_CHECK( hipFree( (void *) d_Rs ),
                   ROCSOLVER_STATUS_ALLOC_ERROR);
        d_Rs = NULL;
        };
    
    };


  /*
   -----------------------------------------------
   prepare to call triangular solvers rf_lusolve()
   -----------------------------------------------
   */




  {

   int const nnz = LUp[n] - LUp[0];


  /*
   ---------------------------------------
   allocate device memory and copy LU data
   ---------------------------------------
   */

  bool const allocate_LUp = !(is_device_pointer( (void *) LUp));
  bool const allocate_LUi = !(is_device_pointer( (void *) LUi));
  bool const allocate_LUx = !(is_device_pointer( (void *) LUx));

  int * d_LUp = (allocate_LUp) ?
                   (int *) rf_mirror_pointer( LUp, sizeof(int)*(n+1) ) :
                   LUp;

  int * d_LUi = (allocate_LUi) ?
                    (int *) rf_mirror_pointer( LUi, sizeof(int)*nnz) :
                    LUi;

  double * d_LUx = (allocate_LUx) ?
                      (double *) rf_mirror_pointer(LUx,sizeof(double)*nnz) :
                      LUx;

   rocsolverStatus_t const istat_lusolve = rf_lusolve( 
                                         handle,
                                         n,
                                         nnz,
                                         d_LUp, 
                                         d_LUi, 
                                         d_LUx,
                                         d_bhat
                                         );
   bool const isok_lusolve = (istat_lusolve == ROCSOLVER_STATUS_SUCCESS );
   if (!isok_lusolve) {
      return( istat_lusolve );
      };

  /*
   ---------------------------------
   release device memory for LU data
   ---------------------------------
   */
  if (allocate_LUp) {
    HIP_CHECK(hipFree( (void *) d_LUp),
              ROCSOLVER_STATUS_ALLOC_ERROR);
    d_LUp = NULL;
  };
  if (allocate_LUi) {
    HIP_CHECK(hipFree((void *) d_LUi),
              ROCSOLVER_STATUS_ALLOC_ERROR);
    d_LUi = NULL;
  };
  if (allocate_LUx) {
    HIP_CHECK(hipFree( (void *) d_LUx),
              ROCSOLVER_STATUS_ALLOC_ERROR);
    d_LUx = NULL;
  };

   };





 
  
  if (need_Q) {
    /*
     -------------------------------
     brhs[ Q_new2old[i] ] = bhat[i]
     -------------------------------
     */
    bool const alloc_Q_new2old = !(is_device_pointer(Q_new2old));
    int * d_Q_new2old = (alloc_Q_new2old) ?
                        (int *) rf_mirror_pointer( Q_new2old, sizeof(int)*n ) :
                        Q_new2old;
 

    rfDipvec( handle, n, d_Q_new2old, d_bhat, d_brhs );


   if (alloc_Q_new2old) {
      HIP_CHECK( hipFree( (void *) d_Q_new2old ),
                 ROCSOLVER_STATUS_ALLOC_ERROR);
      d_Q_new2old = NULL;
   };
 }
 else {
    void *src = (void *) d_bhat;
    void *dest = (void *) d_brhs;
    size_t nbytes = sizeof(double) * n;
    HIP_CHECK( hipMemcpy( dest, src, nbytes, hipMemcpyDeviceToDevice ),
               ROCSOLVER_STATUS_INTERNAL_ERROR);
    };
    

 /*
  --------------------------------
  copy solution from GPU to CPU Host
  --------------------------------
  */

 if (alloc_brhs) {
  void *src = d_brhs;
  void *dest = brhs;
  size_t nbytes = sizeof(double) * n;
  HIP_CHECK( hipMemcpy( dest, src, nbytes, hipMemcpyDeviceToHost ),
             ROCSOLVER_STATUS_INTERNAL_ERROR);


  HIP_CHECK( hipFree( (void *) d_brhs ),
             ROCSOLVER_STATUS_ALLOC_ERROR );
  d_brhs = NULL;
  };

  
  if (d_bhat != NULL) {
     HIP_CHECK( hipFree( (void *) d_bhat ),
                ROCSOLVER_STATUS_ALLOC_ERROR);
     d_bhat = NULL;
     };


 /*
  --------------------------
  clean up handle and stream
  --------------------------
  */

 if (need_handle) {
  hipStream_t stream;
  HIPSPARSE_CHECK( hipsparseGetStream( handle, &stream ), 
                   ROCSOLVER_STATUS_INTERNAL_ERROR );
  HIP_CHECK( hipStreamSynchronize(stream),
             ROCSOLVER_STATUS_INTERNAL_ERROR );
  HIP_CHECK( hipStreamDestroy(stream),
             ROCSOLVER_STATUS_INTERNAL_ERROR );

  HIPSPARSE_CHECK( hipsparseDestroy( handle ),
             ROCSOLVER_STATUS_INTERNAL_ERROR );

  };

  return( ROCSOLVER_STATUS_SUCCESS );
}
 
