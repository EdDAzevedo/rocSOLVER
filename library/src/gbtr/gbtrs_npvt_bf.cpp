
#include "gbtr_common.h"
#include "gbtrs_npvt_bf.hpp"



extern "C" {

void Dgbtrs_npvt_bf
(
hipStream_t stream,

int const nb,
int const nblocks,
int const batchCount,
int const nrhs,
double const * const A_,
int const lda,
double const * const D_,
int const ldd,
double const * const U_,
int const ldu,
double * brhs_,
int const ldbrhs,
int *pinfo
) 
{

  gbtrs_npvt_bf_template<double>
  (
  stream,

  nb,
  nblocks,
  batchCount,
  nrhs,
  A_,
  lda,
  D_,
  ldd,
  U_,
  ldu,
  brhs_,
  ldbrhs,
  pinfo
  );


};



void Zgbtrs_npvt_bf
(
hipStream_t stream,

int const nb,
int const nblocks,
int const batchCount,
int const nrhs,
rocblas_double_complex const * const A_,
int const lda,
rocblas_double_complex const * const D_,
int const ldd,
rocblas_double_complex const * const U_,
int const ldu,
rocblas_double_complex * brhs_,
int const ldbrhs,
int *pinfo
) 
{

  gbtrs_npvt_bf_template<rocblas_double_complex>
  (
  stream,

  nb,
  nblocks,
  batchCount,
  nrhs,
  A_,
  lda,
  D_,
  ldd,
  U_,
  ldu,
  brhs_,
  ldbrhs,
  pinfo
  );


};






}
