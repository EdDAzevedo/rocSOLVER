#include "rocsolver_gbtrsInterleavedBatch.hpp"

extern "C"
{


rocsolverStatus_t  rocsolverDgbtrsInterleavedBatch(
                     rocsolverHandle_t handle,
                     int nb,
                     int nblocks,
                     const double* A_,
                     int lda,
                     const double* B_,
                     int ldb,
                     const double* C_,
                     int ldc,
                     double* brhs_,
                     int ldbrhs,
                     int batchCount
                     )
{


return(
  rocsolver_gbtrsInterleavedBatch_template<double>(
                     handle,
                     nb,
                     nblocks,
                     A_,
                     lda,
                     B_,
                     ldb,
                     C_,
                     ldc,
                     brhs_,
                     ldbrhs,
                     batchCount
                     )
  );
};





rocsolverStatus_t  rocsolverSgbtrsInterleavedBatch(
                     rocsolverHandle_t handle,
                     int nb,
                     int nblocks,
                     const float* A_,
                     int lda,
                     const float* B_,
                     int ldb,
                     const float* C_,
                     int ldc,
                     float* brhs_,
                     int ldbrhs,
                     int batchCount
                     )
{


return(
  rocsolver_gbtrsInterleavedBatch_template<float>(
                     handle,
                     nb,
                     nblocks,
                     A_,
                     lda,
                     B_,
                     ldb,
                     C_,
                     ldc,
                     brhs_,
                     ldbrhs,
                     batchCount
                     )
  );
};










}
