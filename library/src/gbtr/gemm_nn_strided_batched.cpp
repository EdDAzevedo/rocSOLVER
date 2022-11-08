#include "gbtr_common.h"
#include "gemm_nn_strided_batched.hpp"

template< typename T >
void gemm_nn_strided_batched_template
(
hipStream_t stream,

int const m,
int const n,
int const k,
int const batchCount,

T const alpha,

T const * const A_,
int const ldA,
long const strideA,

T const * const B_,
int const ldB,
long const strideB,

T const beta,

T * C_,
int const ldC,
long const strideC
)
{

#ifdef USE_GPU
  hipLaunchKernelGGL( 
    (gemm_nn_strided_batched_kernel),
    dim3( grid_dim ),
    dim3( block_dim ),
    0,
    stream,

    m,n,k,batchCount,
    alpha,  A_,ldA,strideA,  B_,ldB,strideB,
    beta,   C_,ldC,strideC 
    );

#else
  gemm_nn_strided_batched_kernel(
    m,n,k,batchCount,
    alpha,   A_,ldA,strideA,   B_,ldB,strideB,
    beta,    C_,ldC,strideC 
    );
#endif
};



extern "C" {

void Dgemm_nn_strided_batched
(
hipStream_t stream,

int const m,
int const n,
int const k,
int const batchCount,

double const alpha,

double const * const A_,
int const ldA,
long const strideA,

double const * const B_,
int const ldB,
long const strideB,

double const beta,

double * C_,
int const ldC,
long const strideC
)
{

 gemm_nn_strided_batched_template<double>
 (
 stream,
 m,n,k,batchCount,
 alpha,    A_,ldA,strideA,
           B_,ldB,strideB,
 beta,     C_,ldC,strideC 
 );
};




void Zgemm_nn_strided_batched
(
hipStream_t stream,

int const m,
int const n,
int const k,
int const batchCount,

rocblas_double_complex const alpha,

rocblas_double_complex const * const A_,
int const ldA,
long const strideA,

rocblas_double_complex const * const B_,
int const ldB,
long const strideB,

rocblas_double_complex const beta,

rocblas_double_complex * C_,
int const ldC,
long const strideC
)
{

 gemm_nn_strided_batched_template<rocblas_double_complex>
 (
 stream,
 m,n,k,batchCount,
 alpha,    A_,ldA,strideA,
           B_,ldB,strideB,
 beta,     C_,ldC,strideC 
 );
};


}
