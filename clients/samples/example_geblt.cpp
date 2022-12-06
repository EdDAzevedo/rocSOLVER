
#include <algorithm> // for std::min
#include <stdio.h>   // for size_t, printf
#include <vector>
#include <stdlib.h>
#include <assert.h>

#include <hip/hip_runtime.h> // for hip functions
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocblas/rocblas.h> // for all the rocblas C interfaces and type declarations
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations

#include "rocsolver_gebltsv_npvt_strided_batched.hpp"
#include "rocsolver_gebltsv_npvt_batched.hpp"
#include "rocsolver_gebltsv_npvt_interleaved_batch.hpp"

#define H2D(dst,src,nbytes) { CHECK_HIP( hipMemcpyHtoD( \
                               (void *) &( (dst)[0] ),  \
                               (void *) &( (src)[0] ),  \
                               (size_t) (nbytes) ) ) };

#define D2H(dst,src,nbytes) { CHECK_HIP( hipMemcpyDtoH( \
                               (void *) &( (dst)[0] ),  \
                               (void *) &( (src)[0] ),  \
                               (size_t) (nbytes) ) ) };


// Example: Simple example to batched block tridiagonal solver

typedef 
enum {
  STRIDED = 1,
  INTERLEAVED = 2,
  BATCHED = 3
} batch_t;


#define CHECK_HIP( fcn )  { hipError_t istat = (fcn); assert( istat == HIP_SUCCESS ); }



template< batch_t batch_type, typename T, typename I, typename Istride >
void form_geblt(
                        I nb,
                        I nblocks,
                        I nrhs,

                        T* A_array[], T* A_, I lda, Istride strideA,
                        T* B_array[], T* B_, I ldb, Istride strideB,
                        T* C_array[], T* C_, I ldc, Istride strideC,

                        T* X_array[], T* X_, I ldx, Istride strideX,
                        T* Y_array[], T* Y_, I ldy, Istride strideY,
                        I batch_count)
{




auto gindx = [=](I iv, I i, I j, I iblock,  
                 T* A_array[], T* A_, I lda, Istride strideA ) ->  T&  {

       if (batch_type == INTERLEAVED)  {
          // iv + i * batch_count + j * (batch_count * lda)  + 
          //    iblock * (batch_count * lda * nb)
            int64_t ipos =  ((iblock * nb + j)*lda + i) * batch_count + iv;
            return( A_[ ipos ] );
            }
       else if (batch_type == BATCHED) {
           T* Ap = (T *) A_array[ iv ];
           int64_t ipos = (iblock * ((int64_t) nb) + j)*lda + i;
           return(  Ap[ ipos ] );
           }
       else if (batch_type == STRIDED) {
           int64_t ipos =  (iblock * (  (int64_t) nb) + j)*lda + i;
           return( A_[ ipos  + iv * ( (int64_t) strideA)] );
           };
       
   };

auto A = [=](I iv, I i, I j, I iblock) -> T& {
        return( gindx( iv,i,j,iblock,   A_array, A_, lda, strideA ) );
        };

auto B = [=](I iv, I i, I j, I iblock) -> T& {
        return( gindx( iv,i,j,iblock,   B_array, B_, ldb, strideB ) );
        };
  
auto C = [=](I iv, I i, I j, I iblock) -> T& {
        return( gindx( iv,i,j,iblock,   C_array, C_, ldc, strideC ) );
        };

auto X = [=](I iv, I i,  I iblock, I irhs) -> T& {
        return( gindx( iv,i,iblock,irhs,   X_array, X_, ldx, strideX ) );
        };

auto Y = [=](I iv, I i,  I iblock, I irhs) -> T& {
        return( gindx( iv,i,iblock, irhs,    Y_array, Y_, ldy, strideY ) );
        };
  auto drand = [](void) -> double { return( drand48() ); };

   for(I k=0; k < nblocks; k++)  
   for(I j=0; j < nb; j++)  
   for(I i=0; i < nb; i++)  
   for(I iv=0; iv < batch_count; iv++) {
      B(iv,i,j,k) = nb * (2 + drand()); 
      A(iv,i,j,k) = -drand();
      C(iv,i,j,k) = -drand();
      };

 // setup vectors x and y

  for(I irhs=0; irhs < nrhs; irhs++) 
  for(I k=0; k < nblocks; k++) 
  for(I i=0; i < nb; i++) 
  for(I iv=0; iv < batch_count; iv++) {
     I iblock = k;
     X(iv,i,iblock,irhs) = drand();
     Y(iv,i,iblock,irhs) = 0;
     };

 // perform matrix vector multiply

  #pragma omp parallel for collapse(4)
  for(I irhs=0; irhs < nrhs; irhs++) 
  for(I k=0; k < nblocks; k++) 
  for(I i=0; i < nb; i++) 
  for(I iv=0; iv < batch_count; iv++)  {

    for(I j=0; j < nb; j++) {
       Y(iv,i,k,irhs) += B(iv,i,j,k) * X(iv,j,k,irhs);
       if ( (k + 1) < nblocks)  {
          Y(iv,i,k,irhs) += C(iv,i,j,k) * X(iv,j,k+1,irhs);
          };
       if (0 <= (k-1) ) {
          Y(iv,i,k,irhs) += A(iv,i,j,k) * X(iv,j,k-1,irhs);
          };
       };

    };
      

   

};  




template<batch_t batch_type, typename T, typename I, typename Istride>
void test_geblt( rocblas_handle handle, 
                 const I nb, 
                 const I nblocks, 
                 const I nrhs, 
                 const I batch_count) {

  char const * const cbatch_type = 
                      (batch_type == INTERLEAVED) ? "interleaved" :
                      (batch_type == BATCHED)     ? "batched"     :
                      (batch_type == STRIDED)     ? "strided"     :
                      "invalid batch_type";
  printf("%s nb=%d, nblocks=%d, nrhs=%d, batch_count=%d\n",
         cbatch_type, nb, nblocks, nrhs, batch_count );
      

  // allocate storage
  I const lda = (nb + 1);
  I const ldb = (nb + 1);
  I const ldc = (nb + 1);


  std::vector<T> hA_( batch_count * lda * nb * nblocks );
  std::vector<T> hB_( batch_count * ldb * nb * nblocks );
  std::vector<T> hC_( batch_count * ldc * nb * nblocks );

  I const ldx = (nb + 1);
  I const ldy = (nb + 1);

  std::vector<T> hX_( batch_count * (ldx * nblocks * nrhs) );
  std::vector<T> hY_( batch_count * (ldx * nblocks * nrhs) );

  T* dA_ = nullptr;
  T* dB_ = nullptr;
  T* dC_ = nullptr;
  T* dX_ = nullptr;
  T* dY_ = nullptr;

  CHECK_HIP( hipMalloc( &dA_, sizeof(T)*hA_.size() ));
  CHECK_HIP( hipMalloc( &dB_, sizeof(T)*hB_.size() ));
  CHECK_HIP( hipMalloc( &dC_, sizeof(T)*hC_.size() ));

  CHECK_HIP( hipMalloc( &dY_, sizeof(T)*hY_.size() ));
  CHECK_HIP( hipMalloc( &dX_, sizeof(T)*hX_.size() ));

  Istride const strideA = (lda * nb * nblocks );
  Istride const strideB = (ldb * nb * nblocks );
  Istride const strideC = (ldc * nb * nblocks );

  Istride const strideX = (ldx * nblocks * nrhs);
  Istride const strideY = (ldy * nblocks * nrhs);

  std::vector<T*> hA_array(batch_count);
  std::vector<T*> hB_array(batch_count);
  std::vector<T*> hC_array(batch_count);

  std::vector<T*> hX_array(batch_count);
  std::vector<T*> hY_array(batch_count);

  for(I iv=0; iv < batch_count; iv++) {
    hA_array[iv] = dA_ + iv * strideA;
    hB_array[iv] = dB_ + iv * strideB;
    hC_array[iv] = dC_ + iv * strideC;

    hX_array[iv] = dX_ + iv * strideX;
    hY_array[iv] = dY_ + iv * strideY;
    };

  T** dA_array = nullptr;
  T** dB_array = nullptr;
  T** dC_array = nullptr;

  T** dX_array = nullptr;
  T** dY_array = nullptr;

  CHECK_HIP( hipMalloc( &dA_array, sizeof(T*) * hA_array.size() ) );
  CHECK_HIP( hipMalloc( &dB_array, sizeof(T*) * hB_array.size() ) );
  CHECK_HIP( hipMalloc( &dC_array, sizeof(T*) * hC_array.size() ) );

  CHECK_HIP( hipMalloc( &dX_array, sizeof(T*) * hX_array.size() ) );
  CHECK_HIP( hipMalloc( &dY_array, sizeof(T*) * hY_array.size() ) );

  std::vector<I> hinfo_array( batch_count );
  I *dinfo_array = nullptr;
  CHECK_HIP( hipMalloc( &dinfo_array, sizeof(I) * hinfo_array.size() ) );


  if (batch_type == BATCHED) {

    form_geblt<BATCHED>(
                         nb,
                         nblocks,
                         nrhs,
                         dA_array, dA_, lda, strideA,
                         dB_array, dB_, ldb, strideB,
                         dC_array, dC_, ldc, strideC,

                         dX_array, dX_, ldx, strideX,
                         dY_array, dY_, ldy, strideY,
                         batch_count);


                         }
  else if (batch_type == STRIDED) {
    form_geblt<STRIDED>(
                         nb,
                         nblocks,
                         nrhs,
                         dA_array, dA_, lda, strideA,
                         dB_array, dB_, ldb, strideB,
                         dC_array, dC_, ldc, strideC,

                         dX_array, dX_, ldx, strideX,
                         dY_array, dY_, ldy, strideY,
                         batch_count);
        }
  else if (batch_type == INTERLEAVED) {
    form_geblt<INTERLEAVED>(
                         nb,
                         nblocks,
                         nrhs,
                         dA_array, dA_, lda, strideA,
                         dB_array, dB_, ldb, strideB,
                         dC_array, dC_, ldc, strideC,

                         dX_array, dX_, ldx, strideX,
                         dY_array, dY_, ldy, strideY,
                         batch_count);


  };



  
  // copy arrays to GPU device

   H2D( dX_, hX_, hX_.size()*sizeof(T) );
   H2D( dY_, hY_, hY_.size()*sizeof(T) );

   H2D( dA_, hA_, hA_.size()*sizeof(T) );
   H2D( dB_, hB_, hB_.size()*sizeof(T) );
   H2D( dC_, hC_, hC_.size()*sizeof(T) );

   H2D( dA_array, hA_array, hA_array.size()*sizeof(T *) );
   H2D( dB_array, hB_array, hB_array.size()*sizeof(T *) );
   H2D( dC_array, hC_array, hC_array.size()*sizeof(T *) );

  for(auto i=0; i < batch_count; i++) { hinfo_array[i] = 0; };
  H2D( dinfo_array, hinfo_array, hinfo_array.size()*sizeof(I) );

 // perform factorization and solve



  rocblas_status istat = rocblas_status_success;
  if (batch_type == BATCHED) {
     istat = rocsolver_gebltsv_npvt_batched_impl(
              handle, nb, nblocks, nrhs,
              dA_array, lda,  
              dB_array, ldb, 
              dC_array, ldc,
              dY_array, ldy,  
              dinfo_array,
              batch_count );
              
  }
  else if (batch_type == STRIDED) {
     istat = rocsolver_gebltsv_npvt_strided_batched_impl(
              handle, nb, nblocks, nrhs,
              dA_, lda, strideA,
              dB_, ldb, strideB,
              dC_, ldc, strideC,
              dY_, ldy, strideY,
              dinfo_array,
              batch_count );
  }
  else if (batch_type == INTERLEAVED) {
      istat = rocsolver_gebltsv_npvt_interleaved_batch_impl(
                handle, nb, nblocks, nrhs,
                dA_, lda,
                dB_, ldb,
                dC_, ldc,
                dY_, ldy,
                dinfo_array,
                batch_count );
               
  };
  assert( istat == rocblas_status_success );

  D2H( hinfo_array, dinfo_array, hinfo_array.size() * sizeof(I) );
  for(size_t i=0; i < hinfo_array.size(); i++) {
     assert( hinfo_array[i] == 0);
     };

 //  check result

 D2H( hY_, dY_, hY_.size() * sizeof(T) );

 double max_abserr = 0;
 for(size_t i=0; i < hY_.size(); i++) {
   double abserr = std::abs( hX_[i] - hY_[i] );
   max_abserr = std::max( max_abserr, abserr );
   };

 printf("max_abserr = %e\n", max_abserr );

 // clean up
 CHECK_HIP( hipFree( dinfo_array ) );
 CHECK_HIP( hipFree( dX_ ) );
 CHECK_HIP( hipFree( dY_ ) );
 
 CHECK_HIP( hipFree( dA_ ) );
 CHECK_HIP( hipFree( dB_ ) );
 CHECK_HIP( hipFree( dC_ ) );
 CHECK_HIP( hipFree( dA_array ) );
 CHECK_HIP( hipFree( dB_array ) );
 CHECK_HIP( hipFree( dC_array ) );

 
 };
  


  
  



int main() {


  // initialization
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  std::vector<int> nb_table = { 1, 3, 10, 20 };
  std::vector<int> nblocks_table = {1, 10, 128 };
  std::vector<int> batch_count_table = {1, 16 };
  std::vector<int> nrhs_table = {1, 10};

  rocblas_int const nb_table_size = nb_table.size();
  rocblas_int const nblocks_table_size = nblocks_table.size();
  rocblas_int const batch_count_table_size = batch_count_table.size();
  rocblas_int const nrhs_table_size = nrhs_table.size();

  for(rocblas_int inb = 0; inb < nb_table_size; inb++) {
  for(rocblas_int inblocks = 0; inblocks < nblocks_table_size; inblocks++) {
  for(rocblas_int ibatch_count=0; ibatch_count < batch_count_table_size; ibatch_count++) {
  for(rocblas_int inrhs = 0; inrhs < nrhs_table_size; inrhs++) {

      rocblas_int const nb = nb_table[ inb ];
      rocblas_int const nblocks = nblocks_table[ inblocks ];
      rocblas_int const batch_count = batch_count_table[ ibatch_count ];
      rocblas_int const nrhs = nrhs_table[ inrhs ];

#define TEST_GEBLT(batch_type) \
    {  \
     test_geblt<batch_type,double,rocblas_int,rocblas_stride>( \
                 handle, nb,nblocks, nrhs, batch_count); \
     test_geblt<batch_type,float,rocblas_int,rocblas_stride>( \
                 handle, nb,nblocks, nrhs, batch_count); \
     test_geblt<batch_type,rocblas_double_complex,rocblas_int,rocblas_stride>( \
                 handle, nb,nblocks, nrhs, batch_count); \
     test_geblt<batch_type,rocblas_float_complex,rocblas_int,rocblas_stride>( \
                 handle, nb,nblocks, nrhs, batch_count); \
     };

      TEST_GEBLT( INTERLEAVED );	
      TEST_GEBLT( STRIDED );	
      TEST_GEBLT( BATCHED );	
      

    };
    };
    };
    };


  // clean up
  rocblas_destroy_handle(handle);
}
