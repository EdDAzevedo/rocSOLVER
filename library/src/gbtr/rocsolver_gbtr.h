#pragma once
#ifndef ROCSOLVER_GBTR_H
#define ROCSOLVER_GBTR_H




extern "C" {

rocsolverStatus_t  rocsolverDgbtrfInterleavedBatch(
                     rocsolverHandle_t handle,
                     int nb,
                     int nblocks,
                     const double* A_,
                     int lda,
                     const double* B_,
                     int ldb,
                     const double* C_,
                     int ldc,
                     int batchCount
                     );


rocsolverStatus_t  rocsolverSgbtrfInterleavedBatch(
                     rocsolverHandle_t handle,
                     int nb,
                     int nblocks,
                     const float* A_,
                     int lda,
                     const float* B_,
                     int ldb,
                     const float* C_,
                     int ldc,
                     int batchCount
                     );
 

rocsolverStatus_t rocsolverDgbtrsInterleavedBatch(
                     rocsolverHandlt_t handle,
                     int nb,
                     int nblocks,
                     const double* A_,
                     int lda,
                     const double* B_,
                     int ldb,
                     const double* C_,
                     int ldc,
                     double *brhs_,
                     int ldbrhs,
                     int batchCount
                     );



rocsolverStatus_t rocsolverSgbtrsInterleavedBatch(
                     rocsolverHandlt_t handle,
                     int nb,
                     int nblocks,
                     const float* A_,
                     int lda,
                     const float* B_,
                     int ldb,
                     const float* C_,
                     int ldc,
                     float *brhs_,
                     int ldbrhs,
                     int batchCount
                     );
                        



};
#endif
