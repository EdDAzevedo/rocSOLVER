#include "gbtr_common.h"
#include "gbtrf_npvt_bf.hpp"



extern "C" {

void Dgbtrf_npvt_bf(
hipStream_t stream,

int const nb,
int const nblocks,
int const batchCount,
double * A_,
int const lda,
double * B_,
int const ldb,
double * C_,
int const ldc,
int *pinfo) 
{

  gbtrf_npvt_bf_template<double>
  (
  stream,
  nb,
  nblocks,
  batchCount,
  A_,
  lda,
  B_,
  ldb,
  C_,
  ldc,
  pinfo
  );


};



void Zgbtrf_npvt_bf(
hipStream_t stream,
int const nb,
int const nblocks,
int const batchCount,
rocblas_double_complex * A_,
int const lda,
rocblas_double_complex * B_,
int const ldb,
rocblas_double_complex * C_,
int const ldc,
int *pinfo) 
{

  gbtrf_npvt_bf_template<rocblas_double_complex>
  (
  stream,
  nb,
  nblocks,
  batchCount,
  A_,
  lda,
  B_,
  ldb,
  C_,
  ldc,
  pinfo
  );


};



}
