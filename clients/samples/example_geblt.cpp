#include <algorithm> // for std::min
#include <hip/hip_runtime_api.h> // for hip functions
#include <rocsolver/rocsolver.h> // for all the rocsolver C interfaces and type declarations
#include <stdio.h>   // for size_t, printf
#include <vector>
#include <stdlib.h>
#include <assert.h>

// Example: Simple example to batched block tridiagonal solver

typedef 
enum {
  STRIDED = 1,
  INTERLEAVED = 2,
  BATCHED = 3
} batch_t;


#define HIP_CHECK( fcn )  { hipStatus_t istat = (fcn); assert( istat == HIP_SUCCESS ); }


template< typename T, typename I, typename U, batch_t batch_type>
void form_geblt(
                        I nb,
                        I nblocks,
                        U A_, I lda, I strideA,
                        U B_, I ldb, I strideB,
                        U C_, I ldc, I strideC,

                        T* x_, 
                        T* y_, 
                        I batchCount)
{



auto x = [=](I iv, I i, I iblock, I irhs) -> T& {
  
          // iv + i * batchCount + iblock * (batchCount * ldx )  + 
          //    irhs * (batchCount * ldx * nblocks)
          I const ldx = nb;
          int64_t ipos = ((irhs * ((int64_t) nblocks) + iblock) * ldx + i ) * batchCount + iv; 
          return( &(x_[ ipos ]) );
          };
            

auto y = [=](I iv, I i, I iblock, I irhs) -> T& {
  
          // iv + i * batchCount + iblock * (batchCount * ldy )  + 
          //    irhs * (batchCount * ldy * nblocks)
          I const ldy = nb;
          int64_t ipos = ((irhs * ((int64_t) nblocks) + iblock) * ldy + i ) * batchCount + iv; 
          return( &(y_[ ipos ]) );
          };

auto gindx = [=](I iv, I i, I j, I iblock,  
                 U A_, I lda, I strideA ) ->  T&  {

       if (batch_type == INTERLEAVED)  {
          // iv + i * batchCount + j * (batchCount * lda)  + 
          //    iblock * (batchCount * lda * nb)
            int64_t ipos =  ((iblock * nb + j)*lda + i) * batchCount + iv;
            return( &(A_[ ipos ]) );
            }
       else if (batch_type == BATCH) {
           T* Ap = (T *) A_[ iv ];
           int64_t ipos = (iblock * ((int64_t) nb) + j)*lda + i;
           return(  &(Ap[ ipos ]) );
           }
       else if (batch_type == STRIDED) {
           int64_t ipos =  (iblock * (  (int64_t) nb) + j)*lda + i;
           return( &(A_[ ipos  + iv * ( (int64_t) strideA)]) );
           };
       
   };

auto A = [=](I iv, I i, I j, I iblock) -> T& {
        return( gindx( iv,i,j,iblock,   A_, lda, strideA ) );
        };

auto B = [=](I iv, I i, I j, I iblock) -> T& {
        return( gindx( iv,i,j,iblock,   B_, ldb, strideB ) );
        };
  
auto C = [=](I iv, I i, I j, I iblock) -> T& {
        return( gindx( iv,i,j,iblock,   C_, ldc, strideC ) );
        };

  auto drand = (void) -> double { return( drand48() ); };

   for(I k=0; k < nblocks; k++) {
   for(I j=0; j < nb; j++) {
   for(I i=0; i < nb; i++) {
   for(I iv=0; iv < batchCount; iv++) {
      B(iv,i,j,k) = nb * (2 + drand()); 
      A(iv,i,j,k) = -drand();
      C(iv,i,j,k) = -drand();
      };
      };
      };
      };

 // setup vectors x and y

  #pragma omp collapse(4)
  for(I irhs=0; irhs < nrhs; irhs++) 
  for(I k=0; k < nblocks; k++) 
  for(I i=0; i < nb; i++) 
  for(I iv=0; iv < batchCount; iv++) {
     x(iv,i,k,irhs) = drand();
     y(iv,i,k,irhs) = 0;
     };

 // perform matrix vector multiply

  #pragma omp collapse(4)
  for(I irhs=0; irhs < nrhs; irhs++) 
  for(I k=0; k < nblocks; k++) 
  for(I i=0; i < nb; i++) 
  for(I iv=0; iv < batchCount; iv++)  {

    for(I j=0; j < nb; j++) {
       y(iv,i,k,irhs) += B(iv,i,j,k) * x(iv,j,k,irhs);
       if (k < (nblocks-1))  {
          y(iv,i,k,irhs) += C(iv,i,j,k) * x(iv,j,k+1,irhs);
          };
       if (1 <= k) {
          y(iv,i,k,irhs) += A(iv,i,j,k) * x(iv,j,k-1,irhs);
          };
       };

    };
      

   

};  




typename<typename T, typename I, batch_t batch_type>
int test_geblt_strided_batched( I nb, I nblocks, I batchCount) {

  // allocate storage
  I const lda = (nb + 1);
  I const ldb = (nb + 1);
  I const ldc = (nb + 1);

  std::vector<T> hA_( batchCount * lda * nb * nblocks );
  std::vector<T> hB_( batchCount * ldb * nb * nblocks );
  std::vector<T> hC_( batchCount * ldc * nb * nblocks );

  I const nrhs = 1;
  std::vector<T> hx_( batchCount * (nb*nblocks) * nrhs );
  std::vector<T> hbrhs_( batchCount * (nb*nblocks) * nrhs );

  T* dA_ = nullptr;
  T* dB_ = nullptr;
  T* dC_ = nullptr;
  T* dx = nullptr;
  T* dbrhs_ = nullptr;

  HIP_CHECK( hipMalloc( &dA_, sizeof(T)*hA_.size() );
  HIP_CHECK( hipMalloc( &dB_, sizeof(T)*hB_.size() );
  HIP_CHECK( hipMalloc( &dC_, sizeof(T)*hC_.size() );

  HIP_CHECK( hipMalloc( &dbrhs_, sizeof(T)*hbrhs_.size() );
  HIP_CHECK( hipMalloc( &dx_, sizeof(T)*hx_.size() );

  I const strideA = (lda * nb * nblocks );
  I const strideB = (ldb * nb * nblocks );
  I const strideC = (ldc * nb * nblocks );

  std::vector<T*> hA_array;
  std::vector<T*> hB_array;
  std::vector<T*> hC_array;

  for(I iv=0; iv < batchCount; iv++) {
    hA_array[iv] = dA_ + iv * strideA;
    hB_array[iv] = dB_ + iv * strideB;
    hC_array[iv] = dC_ + iv * strideC;
    };

  T** dA_array = nullptr;
  T** dB_array = nullptr;
  T** dC_array = nullptr;

  HIP_CHECK( hipMalloc( &dA_array, sizeof(T*) * hA_array.size() ) );
  HIP_CHECK( hipMalloc( &dB_array, sizeof(T*) * hB_array.size() ) );
  HIP_CHECK( hipMalloc( &dC_array, sizeof(T*) * hC_array.size() ) );



  if (batch_type == BATCHED) {

    form_geblt<T,I,T**,BATCHED>(
                         nb,
                         nblocks,
                         dA_array, lda, strideA,
                         dB_array, ldb, strideB,
                         dC_array, ldc, strideC,

                         dx_, 
                         dbrhs_, 
                         batchCount);


                         }
  else if (batch_type == STRIDED) {
    form_geblt<T,I,T*,STRIDED>(
                         nb,
                         nblocks,
                         dA_, lda, strideA,
                         dB_, ldb, strideB,
                         dC_, ldc, strideC,

                         dx_, 
                         dbrhs_, 
                         batchCount);
        }
  else if (batch_type == INTERLEAVED) {
    form_geblt<T,I,T*,INTERLEAVED>(
                         nb,
                         nblocks,
                         dA_, lda, strideA,
                         dB_, ldb, strideB,
                         dC_, ldc, strideC,

                         dx_, 
                         dbrhs_, 
                         batchCount);


  };

  
  // copy arrays to GPU device
  HIP_CHECK( hipMemcpyH2D( dx_, hx_, hx_.size()*sizeof(T) ) );
  HIP_CHECK( hipMemcpyH2D( dbrhs_, hbrhs_, hbrhs_.size()*sizeof(T) ) );

  HIP_CHECK( hipMemcpyH2D( dA_, hA_, hA_.size()*sizeof(T) ) );
  HIP_CHECK( hipMemcpyH2D( dB_, hB_, hB_.size()*sizeof(T) ) );
  HIP_CHECK( hipMemcpyH2D( dC_, hC_, hC_.size()*sizeof(T) ) );

  HIP_CHECK( hipMemcpyH2D( dA_array, hA_array, hA_array.size()*sizeof(T *) ) );
  HIP_CHECK( hipMemcpyH2D( dB_array, hB_array, hB_array.size()*sizeof(T *) ) );
  HIP_CHECK( hipMemcpyH2D( dC_array, hC_array, hC_array.size()*sizeof(T *) ) );




  
  
};
  



int main() {


  // initialization
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // calculate the sizes of our arrays
  size_t size_A = size_t(lda) * N;          // count of elements in matrix A
  size_t size_piv = size_t(std::min(M, N)); // count of Householder scalars

  // allocate memory on GPU
  double *dA, *dIpiv;
  hipMalloc(&dA, sizeof(double)*size_A);
  hipMalloc(&dIpiv, sizeof(double)*size_piv);

  // copy data to GPU
  hipMemcpy(dA, hA.data(), sizeof(double)*size_A, hipMemcpyHostToDevice);

  // compute the QR factorization on the GPU
  rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv);

  // copy the results back to CPU
  std::vector<double> hIpiv(size_piv); // array for householder scalars on CPU
  hipMemcpy(hA.data(), dA, sizeof(double)*size_A, hipMemcpyDeviceToHost);
  hipMemcpy(hIpiv.data(), dIpiv, sizeof(double)*size_piv, hipMemcpyDeviceToHost);

  // the results are now in hA and hIpiv
  // we can print some of the results if we want to see them
  printf("R = [\n");
  for (size_t i = 0; i < M; ++i) {
    printf("  ");
    for (size_t j = 0; j < N; ++j) {
      printf("% .3f ", (i <= j) ? hA[i + j*lda] : 0);
    }
    printf(";\n");
  }
  printf("]\n");

  // clean up
  hipFree(dA);
  hipFree(dIpiv);
  rocblas_destroy_handle(handle);
}
