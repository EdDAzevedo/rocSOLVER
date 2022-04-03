
#ifndef SYNCTHREADS

#ifdef USE_CPU
#define SYNCTHREADS() {}
#else
#define SYNCTHREADS() { __syncthreads(); }
#endif

#endif
