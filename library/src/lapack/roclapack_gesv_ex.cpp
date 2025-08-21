/* **************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "roclapack_gesv_ex.hpp"
#include "rocblas_utility.hpp"
#include <rocblas/internal/rocblas-types.h>

#include "roclapack_gesv.hpp"
#include "roclapack_getrf.hpp"
#include "roclapack_getrf_mxp.hpp"
#include "roclapack_getrs.hpp"

ROCSOLVER_BEGIN_NAMESPACE

// The following traits enforce that homogenous computation is viable if
//  - A, B, X, compute_type are all the same type
//  - A, B, X, compute type are lapack SDCZ storage types

template <typename T>
constexpr bool is_gesv_ex_homogenous_storage
    = std::is_same_v<
          T,
          float> || std::is_same_v<T, double> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>;

template <typename T, typename... Ts>
constexpr bool gesv_ex_homogenous_accepts = (std::is_same_v<T, Ts> && ...)
    && (is_gesv_ex_homogenous_storage<T> && ... && is_gesv_ex_homogenous_storage<Ts>);

template <typename T>
rocblas_status rocsolver_gesv_ex_homogenous(rocblas_handle handle,
                                            const rocblas_int n,
                                            const rocblas_int nrhs,
                                            T* A,
                                            const rocblas_int lda,
                                            rocblas_int* ipiv,
                                            T* B,
                                            const rocblas_int ldb,
                                            T* X,
                                            const rocblas_int ldx,
                                            const rocblas_int max_iter,
                                            const double tol,
                                            rocblas_int* niter,
                                            rocblas_int* info)
{
    // XXX: this is just the body of gesv_impl from roclapack_gesv.cpp, it is not adapted for explicit X, etc.
    using S = decltype(std::real(T{}));

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st = rocsolver_gesv_argCheck(handle, n, nrhs, lda, ldb, A, B, ipiv, info);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    rocblas_int shiftA = 0;
    rocblas_int shiftB = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideB = 0;
    rocblas_stride strideP = 0;
    rocblas_int batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of reusable workspace (and for calling GETRF and GETRS)
    bool optim_mem;
    size_t size_work, size_work1, size_work2, size_work3, size_work4;
    // extra requirements for calling GETRF
    size_t size_pivotval, size_pivotidx, size_iinfo, size_iipiv;
    rocsolver_gesv_getMemorySize<false, false, T>(
        n, nrhs, batch_count, &size_scalars, &size_work, &size_work1, &size_work2, &size_work3,
        &size_work4, &size_pivotval, &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_set_optimal_device_memory_size(
            handle, size_scalars, size_work, size_work1, size_work2, size_work3, size_work4,
            size_pivotval, size_pivotidx, size_iipiv, size_iinfo);
    // memory workspace allocation
    void *scalars, *work, *work1, *work2, *work3, *work4, *pivotval, *pivotidx, *iinfo, *iipiv;
    rocblas_device_malloc mem(handle, size_scalars, size_work, size_work1, size_work2, size_work3,
                              size_work4, size_pivotval, size_pivotidx, size_iipiv, size_iinfo);

    if(!mem)
        return rocblas_status_memory_error;

    scalars = mem[0];
    work = mem[1];
    work1 = mem[2];
    work2 = mem[3];
    work3 = mem[4];
    work4 = mem[5];
    pivotval = mem[6];
    pivotidx = mem[7];
    iipiv = mem[8];
    iinfo = mem[9];
    if(size_scalars > 0)
        init_scalars(handle, (T*)scalars);

    // execution
    return rocsolver_gesv_template<false, false, T>(
        handle, n, nrhs, A, shiftA, lda, strideA, ipiv, strideP, B, shiftB, ldb, strideB, info,
        batch_count, (T*)scalars, (T*)work, work1, work2, work3, work4, (T*)pivotval,
        (rocblas_int*)pivotidx, (rocblas_int*)iipiv, (rocblas_int*)iinfo, optim_mem);
}

template <typename TA, typename TB, typename TX, typename Tc, typename...>
struct gesv_homogenous_call
{
    rocblas_status operator()(rocblas_handle handle,
                              const rocblas_int n,
                              const rocblas_int nrhs,
                              void* A,
                              const rocblas_int lda,
                              rocblas_int* ipiv,
                              void* B,
                              const rocblas_int ldb,
                              void* X,
                              const rocblas_int ldx,
                              const rocblas_int max_iter,
                              const double tol,
                              rocblas_int* niter,
                              rocblas_int* info)
    {
        if constexpr(gesv_ex_homogenous_accepts<TA, TB, TX, Tc>)
        {
            return rocsolver_gesv_ex_homogenous(handle, n, nrhs, (TA*)A, lda, ipiv, (TB*)B, ldb,
                                                (TX*)X, ldx, max_iter, tol, niter, info);
        }

        return rocblas_status_not_implemented;
    }
};

// -----------------------------------------------------
// gather the value from iamax into xnrm
//
// assume launch as dim3( nbx, 1, nbz ), dim3(nx,1,1)
// nbx = ceil( nrhs, nx )
// nby = batch_count
// -----------------------------------------------------
template <typename T, typename S>
__device__ static void gather_norm_kernel(rocblas_int const n,
                                          rocblas_int const nrhs,

                                          T* const X,
                                          rocblas_stride const shiftX,
                                          rocblas_int const ldx,
                                          rocblas_stride const strideX,

                                          rocblas_int* const ixnrm,

                                          S* const xnrm,

                                          rocblas_int const batch_count)
{
    rocblas_int const bid_start = blockIdx.z;
    rocblas_int const bid_inc = gridDim.z;

    rocblas_int const irhs_start = threadIdx.x + blockIdx.x * blockDim.x;
    rocblas_int const irhs_inc = blockDim.x * gridDim.x;

    auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

    for(rocblas_int irhs = irhs_start; irhs < nrhs; irhs += irhs_inc)
    {
        for(rocblas_int bid = bid_start; bid < batch_count; bid += bid_inc)
        {
            auto const Xp = load_ptr_batch(X, bid, shiftX, strideX);

            auto const irow = ixnrm[bid + irhs * batch_count];
            auto const jcol = irhs;
            auto const xi = Xp[idx2D(irow, jcol, ldx)];
            xnrm[bid + irhs * batch_count] = std::abs(xi);
        }
    }
}

template <typename T, typename S>
static void gather_norm(rocblas_handle handle,
                        rocblas_int const n,
                        rocblas_int const nrhs,

                        T* const X,
                        rocblas_stride const shiftX,
                        rocblas_int const ldx,
                        rocblas_stride const strideX,

                        rocblas_int* const ixnrm,

                        S* const xnrm,

                        rocblas_int const batch_count)
{
    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    rocblas_int const max_blocks = 1024;
    rocblas_int const nx = 64;
    rocblas_int const nbx = std::min(max_blocks, ceil(n, nx));
    rocblas_int const nby = 1;
    rocblas_int const nbz = std::min(max_blocks, batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    gather_norm_kernel<T, S><<<dim3(nbx, nby, nbz), dim3(nx, 1, 1), 0, stream>>>(

        n, nrhs,

        X, shiftX, ldx, strideX,

        ixnrm, xnrm, batch_count);
}

// -----------------------------------------
// assume one thread block handle one vector
// launch as
// dim3(1,nrhs,batch_count), dim3(nx,1,1,), ldsize, stream
// ldsize = nx * sizeof(double)
// -----------------------------------------
template <typename T, typename I, typename Istride>
__global__ static void check_convergence_kernel(I const n,
                                                I const nrhs,

                                                T* X,
                                                Istride const shiftX,
                                                I const ldx,
                                                Istride const strideX,

                                                T* R,
                                                Istride const shiftR,
                                                I const ldr,
                                                Istride const strideR,

                                                I const batch_count,
                                                double tol,

                                                int* p_is_converged)
{
    extern __shared__ double ldmem[];

    I const bid_start = blockIdx.z;
    I const bid_inc = gridDim.z;

    I const irhs_start = blockIdx.y;
    I const irhs_inc = gridDim.y;

    I const tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
    I const nthreads = (blockDim.x * blockDim.y) * blockDim.z;

    I const i_start = tid;
    I const i_inc = nthreads;

    assert(gridDim.x == 1);

    I non_converged = 0;
    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const Xp = load_ptr_batch(X, bid, shiftX, strideX);
        T const* const Rp = load_ptr_batch(R, bid, shiftR, strideR);

        for(I irhs = irhs_start; irhs < nrhs; irhs += irhs_inc)
        {
            double xmax = 0;
            double rmax = 0;
            for(I i = i_start; i < n; i += i_inc)
            {
                auto const xi = Xp[idx2D(i, irhs, ldx)];
                auto const ri = Rp[idx2D(i, irhs, ldr)];
                double const abs_xi = std::abs(xi);
                double const abs_ri = std::abs(ri);

                xmax = std::max(xmax, abs_xi);
                rmax = std::max(rmax, abs_ri);
            }
            __syncthreads();

            // ---------------------------------
            // perform max reduction
            // the answer is in the [0] position
            // ---------------------------------
            auto max_reduce = [=](auto& xmax) {
                ldmem[tid] = xmax;
                __syncthreads();

                for(I gap = nthreads / 2; gap > 0; gap = gap / 2)
                {
                    if(tid < gap)
                    {
                        ldmem[tid] = std::max(ldmem[tid], ldmem[tid + gap]);
                    }
                    __syncthreads();
                }
                __syncthreads();
                xmax = ldmem[0];
                __syncthreads();
            };

            max_reduce(xmax);
            max_reduce(rmax);

            if(tid == 0)
            {
                if(rmax > xmax * tol)
                {
                    non_converged++;
                }
            }

        } // end for irhs
        __syncthreads();
    } // end for bid

    __syncthreads();

    if(tid == 0)
    {
        if(non_converged >= 1)
        {
            // -------------------------
            // set  is_converged to false
            // -------------------------
            atomicMin(p_is_converged, 0);
        }
    }
}

// check for convergence
template <typename T>
static void check_convergence(rocblas_handle handle,
                              rocblas_int const n,
                              rocblas_int const nrhs,

                              T* X,
                              rocblas_stride const shiftX,
                              rocblas_int const ldx,
                              rocblas_stride const strideX,

                              T* R,
                              rocblas_stride const shiftR,
                              rocblas_int const ldr,
                              rocblas_stride const strideR,

                              rocblas_int const batch_count,
                              double tol,

                              int* d_is_converged)
{
    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_int const nx = 1024;
    size_t const ldsize = sizeof(double) * nx;

    {
        // ------------------------------------
        // set initial value for d_is_converged
        // ------------------------------------
        int is_converged = true;
        auto const istat = hipMemcpyAsync(d_is_converged, &is_converged, sizeof(int),
                                          hipMemcpyHostToDevice, stream);
        assert(istat == hipSuccess);
    }

    check_convergence_kernel<T><<<dim3(1, nrhs, batch_count), dim3(nx, 1, 1), ldsize, stream>>>(
        n, nrhs,

        X, shiftX, ldx, strideX,

        R, shiftR, ldr, strideR,

        batch_count, tol, d_is_converged);
}

// The following traits allow selecting a reduced precision companion type for a given compute type

template <typename T>
struct gesv_ex_mxp_lu_reduced_precision
{
    using type = void;
};

template <>
struct gesv_ex_mxp_lu_reduced_precision<float>
{
    using type = rocblas_half;
};

template <typename T>
using gesv_ex_mxp_lu_reduced_precision_t = typename gesv_ex_mxp_lu_reduced_precision<T>::type;

// The following traits enforce that MXP LU computation is viable if:
//  - A, B, X are all the same type
//  - A, B, X are lapack SDCZ storage types
//  - compute_type is SDCZ, half, or bfloat16

template <typename T>
constexpr bool is_gesv_ex_mxp_lu_storage
    = std::is_same_v<
          T,
          float> || std::is_same_v<T, double> || std::is_same_v<T, rocblas_float_complex> || std::is_same_v<T, rocblas_double_complex>;

template <typename T>
constexpr bool is_gesv_ex_mxp_lu_compute
    = // std::is_same_v<T, rocblas_half> || std::is_same_v<T, rocblas_bfloat16> ||
    is_gesv_ex_mxp_lu_storage<T>;

template <typename TA, typename TB, typename TX, typename Tc>
constexpr bool gesv_ex_mxp_lu_accepts
    = (std::is_same_v<TA, TB> && std::is_same_v<TA, TX>)&&is_gesv_ex_mxp_lu_storage<
        TA>&& is_gesv_ex_mxp_lu_compute<Tc>;

template <typename T, typename LU>
rocblas_status rocsolver_gesv_ex_mxp_lu(rocblas_handle handle,
                                        const rocblas_int n,
                                        const rocblas_int nrhs,
                                        T* A,
                                        const rocblas_stride shiftA,
                                        const rocblas_int lda,
                                        const rocblas_stride strideA,
                                        rocblas_int* ipiv,
                                        const rocblas_stride strideP,
                                        T* B,
                                        const rocblas_stride shiftB,
                                        const rocblas_int ldb,
                                        const rocblas_stride strideB,
                                        T* X,
                                        const rocblas_stride shiftX,
                                        const rocblas_int ldx,
                                        const rocblas_stride strideX,
                                        const rocblas_int max_iter_arg,
                                        const double tol_arg,
                                        rocblas_int* niter,
                                        rocblas_int* info,
                                        const rocblas_int batch_count,
                                        const bool use_pivot)
{
    ROCSOLVER_ENTER("gesv_ex_mxp_lu", "n:", n, "nrhs:", nrhs, "shiftA", shiftA, "lda:", lda,
                    "strideA", strideA, "strideP", strideP, "shiftB", shiftB, "ldb:", ldb,
                    "strideB", strideB, "shiftX", shiftX, "ldx", ldx, "strideX", strideX, "tol",
                    tol_arg, "max_iter", max_iter_arg);

    *niter = 0;
    *info = 0;

    bool constexpr is_complex = rocblas_is_complex<T>;

    using Sfull = typename std::conditional<is_complex, decltype(std::real(T{})), T>::type;
    using Slu = decltype(std::real(LU{}));
    using Sreduced = decltype(std::real(LU{}));

    double const tol_default = std::numeric_limits<Sfull>::epsilon() * n;

    double const tol = (tol_arg <= 0) ? tol_default : tol_arg;

    rocblas_int const max_iter_default = 30;
    rocblas_int const max_iter = (max_iter_arg <= 0) ? max_iter_default : max_iter_arg;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // XXX: can we use std::ceil/ceilf?
    auto ceil = [](auto n, auto b) { return ((n - 1) / b + 1); };

    // ----------
    // reset info
    // ----------

    ROCSOLVER_LAUNCH_KERNEL(reset_info, dim3(ceil(batch_count, BS1), 1, 1), dim3(BS1, 1, 1), 0,
                            stream, info, batch_count, 0);

    void* work = nullptr;
    size_t size_work = 0;
    std::byte* const pwork = (std::byte*)work;
    std::byte* pfree = pwork;

#ifndef CHECK_MEM
#define CHECK_MEM(pfree)                                       \
    {                                                          \
        bool const is_mem_ok = (pfree <= (pwork + size_work)); \
        if(!is_mem_ok)                                         \
        {                                                      \
            return (rocblas_status_memory_error);              \
        }                                                      \
    }
#endif

    // ----------------------
    // allocate B_lu and A_lu
    // ----------------------

    auto const nrows_A = n;
    auto const ncols_A = n;

    auto const nrows_B = n;
    auto const ncols_B = nrhs;

    auto const nrows_X = nrows_B;
    auto const ncols_X = ncols_B;

    rocblas_int const ldB_lu = nrows_B;
    rocblas_stride const strideB_lu = ldB_lu * ncols_B;
    size_t const size_B_lu = sizeof(LU) * strideB_lu * batch_count;
    LU* const B_lu = (LU*)pfree;
    pfree += size_B_lu;

    rocblas_int const ldA_lu = nrows_A;
    rocblas_stride const strideA_lu = ldA_lu * ncols_A;
    size_t const size_A_lu = sizeof(LU) * strideA_lu * batch_count;
    LU* const A_lu = (LU*)pfree;
    pfree += size_A_lu;

    CHECK_MEM(pfree);

    rocblas_stride const shiftA_lu = 0;
    rocblas_stride const shiftB_lu = 0;

    // ----------------
    // copy B into B_lu
    // copy A into A_lu
    // ----------------
    {
        char const uplo = 'A';
        // XXX: use standard copying kernels
        // lacpy<rocblas_int, Istride>(handle, uplo, nrows_A, ncols_A,
        //                   A, shiftA, lda, strideA,
        //                   A_lu, shiftA_lu, ldA_lu, strideA_lu,
        //                   batch_count);

        // lacpy<I, Istride>(handle, uplo, nrows_B, ncols_B,
        //                   B, shiftB, ldb, strideB,
        //                   B_lu, shiftB_lu, ldB_lu, strideB_lu,
        //                   batch_count);
    }

    // ------------------------
    // perform LU factorization
    // ------------------------

    bool const use_mixed_precision = !std::is_same<T, LU>::value;
    if(use_mixed_precision)
    {
        auto const pfree_saved = pfree;

        size_t const size_remain = (pwork + size_work) - pfree;

        rocblas_int const inca = 1;

        rocblas_stride shiftP = 0;
        bool const use_pivot = true;

        auto const istat = rocsolver_getrf_mxp_template<LU, LU>(
            handle, nrows_A, ncols_A, A_lu, shiftA_lu, inca, ldA_lu, strideA_lu, ipiv, shiftP,
            strideP, info, batch_count, use_pivot, pfree, size_remain);

        if(istat != rocblas_status_success)
        {
            return (istat);
        }

        pfree = pfree_saved;
    }
    else
    {
        auto const pfree_saved = pfree;

        // ------------------------------------------------------
        // use regular LU factorization (without mixed precision)
        // ------------------------------------------------------

        bool constexpr BATCHED = true;
        bool constexpr STRIDED = true;

        rocblas_int const inca = 1;
        rocblas_stride const shiftP = 0;

        size_t size_scalars = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;
        size_t size_pivotval = 0;
        size_t size_pivotidx = 0;
        size_t size_iipiv = 0;
        size_t size_iinfo = 0;
        bool optim_mem = true;

        rocsolver_getrf_getMemorySize<BATCHED, STRIDED, LU>(
            n, n, use_pivot, batch_count, &size_scalars, &size_work1, &size_work2, &size_work3,
            &size_work4, &size_pivotval, &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

        // XXX: this is too opaque, should be an array or struct
        LU* const scalars = (LU*)pfree;
        pfree += size_scalars;
        LU* const work1 = (LU*)pfree;
        pfree += size_work1;
        LU* const work2 = (LU*)pfree;
        pfree += size_work2;
        LU* const work3 = (LU*)pfree;
        pfree += size_work3;
        LU* const work4 = (LU*)pfree;
        pfree += size_work4;
        LU* const pivotval = (LU*)pfree;
        pfree += size_pivotval;
        rocblas_int* const pivotidx = (rocblas_int*)pfree;
        pfree += size_pivotidx;
        rocblas_int* const iipiv = (rocblas_int*)pfree;
        pfree += size_iipiv;
        rocblas_int* const iinfo = (rocblas_int*)pfree;
        pfree += size_iinfo;

        CHECK_MEM(pfree);

        rocsolver_getrf_template<BATCHED, STRIDED, LU>(
            handle, n, n, A_lu, shiftA_lu, inca, ldA_lu, strideA_lu, ipiv, shiftP, strideP, info,
            batch_count, scalars, work1, work2, work3, work4, pivotval, pivotidx, iipiv, iinfo,
            optim_mem, use_pivot);

        pfree = pfree_saved;
    }

    // ------------------------------------
    // solve the system  A_lu * X_lu = B_lu,
    // where X_lu over-write B_lu
    // ------------------------------------

    // XXX: this is probably too big for a lambda
    auto solve_rhs = [=]() {
        auto pfree_local = pfree;

        rocblas_int const inca = 1;
        rocblas_int const incb = 1;

        rocblas_operation const trans = rocblas_operation_none;

        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;
        bool optim_mem = true;

        bool constexpr BATCHED = true;
        bool constexpr STRIDED = true;
        rocsolver_getrs_getMemorySize<BATCHED, STRIDED, LU>(trans, n, nrhs, batch_count,

                                                            &size_work1, &size_work2, &size_work3,
                                                            &size_work4, &optim_mem);

        LU* const work1 = (LU*)pfree_local;
        pfree_local += size_work1;
        LU* const work2 = (LU*)pfree_local;
        pfree_local += size_work2;
        LU* const work3 = (LU*)pfree_local;
        pfree_local += size_work3;
        LU* const work4 = (LU*)pfree_local;
        pfree_local += size_work4;

        CHECK_MEM(pfree_local);

        return (rocsolver_getrs_template<BATCHED, STRIDED, LU>(
            handle, trans, n, nrhs, A_lu, shiftA_lu, inca, ldA_lu, strideA_lu, ipiv, strideP, B_lu,
            shiftB_lu, incb, ldB_lu, strideB_lu, batch_count, work1, work2, work3, work4, optim_mem,
            use_pivot));
    }; // end solve_rhs()

    {
        auto const istat = solve_rhs();
        if(istat != rocblas_status_success)
        {
            return (istat);
        }
    }

    // --------------------
    // convert solution back to T
    // --------------------
    {
        char const uplo = 'A';
        // XXX: use standard kernels

        // lacpy(handle, uplo, nrows_B, ncols_B,
        //       B_lu, shiftB_lu, ldB_lu, strideB_lu,
        //       X, shiftX, ldx, strideX,
        //       batch_count);
    }
    // ---------------------
    // compute R = B - A * X
    // (1) R <- B
    // (2) R <- R - A * X
    // ---------------------
    rocblas_int const nrows_R = nrows_B;
    rocblas_int const ncols_R = ncols_B;

    rocblas_int const ldr = n;
    rocblas_stride const strideR = ldr * ncols_R;
    rocblas_stride const shiftR = 0;

    size_t const size_R = sizeof(T) * strideR * batch_count;
    T* const R = (T*)pfree;
    pfree += size_R;

    CHECK_MEM(pfree);

    // ----------------------------------------------
    // compute residual using the latest version of X
    // the residual matrix R will be updated
    // ----------------------------------------------
    auto compute_residual = [=] {
        // ----------
        // (1) R <- B
        // ----------
        {
            char const uplo = 'A';
            // lacpy(handle, uplo, n, nrhs,
            //       B, shiftB, ldb, strideB,
            //       R, shiftR, ldr, strideR,
            //       batch_count);
        }

        // ------------------
        // (2) R <- R - A * X
        // ------------------
        {
            T alpha = -1;
            T beta = 1;

            rocblas_int const mm = nrows_R;
            rocblas_int const nn = ncols_R;
            rocblas_int const kk = ncols_A;

            rocblas_operation const trans1 = rocblas_operation_none;
            rocblas_operation const trans2 = rocblas_operation_none;

            rocblas_status istat = rocblasCall_gemm<T>(
                handle, trans1, trans2, mm, nn, kk, &alpha, A, shiftA, lda, strideA, X, shiftX, ldx,
                strideX, &beta, R, shiftR, ldr, strideR, batch_count, (T**)pfree);

            assert(istat == rocblas_status_success);
        }
    }; // end compute_residual()

    compute_residual();

    rocblas_stride iter = 0;
    int is_all_converged = false;

    {
        int* const d_is_all_converged = (int*)pfree;
        pfree += sizeof(int);

        CHECK_MEM(pfree);

        check_convergence(handle, n, nrhs, X, shiftX, ldx, strideX, R, shiftR, ldr, strideR,
                          batch_count, tol, d_is_all_converged);

        HIP_CHECK(hipMemcpyAsync(&is_all_converged, d_is_all_converged, sizeof(int),
                                 hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));

        pfree = pfree - sizeof(int);
    }

    if(is_all_converged)
    {
        *niter = iter;
        *info = 0;
        return (rocblas_status_success);
    }

    for(rocblas_int iter = 0; iter < max_iter; iter++)
    {
        // ---------------------------
        // convert R from FP64 to FP32
        // ---------------------------
        {
            char const uplo = 'A';
            lacpy(handle, uplo, nrows_R, ncols_R,

                  R, shiftR, ldr, strideR,

                  B_lu, shiftB_lu, ldB_lu, strideB_lu,

                  batch_count);
        }

        // -------------------------
        // solve for "dx" correction
        // answer over-writes B_lu
        // -------------------------
        {
            auto const istat = solve_rhs();
            if(istat != rocblas_status_success)
            {
                return (istat);
            }
        }

        // ------------
        // update X <-  X + dx
        // dx is stored in B_lu
        // ------------
        auto update_X = [=] {
            // ------------------
            // update X <- X + dx
            // (1) R <- dx
            // (2) X <- X + R
            // ------------------
            {
                // -----------
                // (1) R <- dx
                // -----------
                char const uplo = 'A';
                lacpy(handle, uplo, nrows_R, ncols_R,

                      B_lu, shiftB_lu, ldB_lu, strideB_lu,

                      R, shiftR, ldr, strideR,

                      batch_count);
            }

            {
                // --------------
                // (2) X <- X + R
                // --------------
                rocblas_stride stride_alpha = 0;
                T alpha = 1;
                rocblas_int const inc1 = 1;
                rocblas_int const inc2 = 1;

                // XXX: this should probably be a bespoke kernel
                for(rocblas_int irhs = 0; irhs < nrhs; irhs++)
                {
                    rocblas_status istat = rocblas_status_success;
                    // auto const istat
                    //     = rocblasCall_axpy(handle,
                    //                        n, &alpha, stride_alpha,
                    //                        R, shiftR, inc1, strideR,
                    //                        X + idx2D(0, irhs, ldx), shiftX, inc2, strideX,
                    //                        batch_count);
                    assert(istat == rocblas_status_success);
                } // end for irhs
            }
        }; // end update_X()

        update_X();

        // ---------------
        // compute residual R
        // using latest version of X
        // ---------------
        compute_residual();

        // ---------
        // B_lu <- R
        // ---------
        {
            char const uplo = 'A';
            lacpy(handle, uplo, nrows_B, ncols_B,

                  R, shiftR, ldr, strideR,

                  B_lu, shiftB_lu, ldB_lu, strideB_lu,

                  batch_count);
        }

        // -----------------------
        // compute correction "dx"
        // dx over-writes B_lu
        // -----------------------
        solve_rhs();

        // -----------------
        // check convergence
        // -----------------

        int is_all_converged = false;

        {
            int* const d_is_all_converged = (int*)pfree;
            pfree += sizeof(int);

            CHECK_MEM(pfree);

            check_convergence(handle, n, nrhs,

                              X, shiftX, ldx, strideX,

                              R, shiftR, ldr, strideR,

                              batch_count, tol, d_is_all_converged);

            HIP_CHECK(hipMemcpyAsync(&is_all_converged, d_is_all_converged, sizeof(int),
                                     hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            pfree = pfree - sizeof(int);
        }

        if(is_all_converged)
        {
            *info = 0;
            *niter = iter;
            return (rocblas_status_success);
        }

    } // for iter

    //  ----------------------------------------------------------
    //  iterative refinement using LU in FP32 with mixed precision
    //  was not able to converge
    //  ----------------------------------------------------------

    *niter = -(max_iter + 1);

    // --------------
    // reset workspace
    // --------------
    pfree = pwork;

    {
        bool constexpr BATCHED = true;
        bool constexpr STRIDED = true;

        size_t size_scalars = 0;
        size_t size_work0 = 0;
        size_t size_work1 = 0;
        size_t size_work2 = 0;
        size_t size_work3 = 0;
        size_t size_work4 = 0;

        size_t size_pivotval = 0;

        size_t size_pivotidx = 0;
        size_t size_iipiv = 0;
        size_t size_iinfo = 0;
        bool optim_mem = true;

        rocsolver_gesv_getMemorySize<BATCHED, STRIDED, T>(
            n, nrhs, batch_count,

            &size_scalars, &size_work0, &size_work1, &size_work2, &size_work3, &size_work4,

            &size_pivotval,

            &size_pivotidx, &size_iipiv, &size_iinfo, &optim_mem);

        T* const scalars = (T*)pfree;
        pfree += size_scalars;

        T* const work0 = (T*)pfree;
        pfree += size_work0;

        T* const work1 = (T*)pfree;
        pfree += size_work1;

        T* const work2 = (T*)pfree;
        pfree += size_work2;

        T* const work3 = (T*)pfree;
        pfree += size_work3;

        T* const work4 = (T*)pfree;
        pfree += size_work4;

        T* const pivotval = (T*)pfree;
        pfree += size_pivotval;

        rocblas_int* const pivotidx = (rocblas_int*)pfree;
        pfree += size_pivotidx;

        rocblas_int* const iipiv = (rocblas_int*)pfree;
        pfree += size_iipiv;

        rocblas_int* const iinfo = (rocblas_int*)pfree;
        pfree += size_iinfo;

        CHECK_MEM(pfree);

        // ------
        // X <- B
        // ------
        {
            auto const uplo = 'A';
            lacpy(handle, uplo, nrows_B, ncols_B,

                  B, shiftB, ldb, strideB,

                  X, shiftX, ldx, strideX,

                  batch_count);
        }

        // XXX: not sure what this is doing
        {
            auto const istat = rocsolver_gesv_template<BATCHED, STRIDED, T>(
                handle, n, nrhs,

                A, shiftA, lda, strideA,

                ipiv, strideP,

                X, shiftX, ldx, strideX,

                info, batch_count,

                scalars, work0, work1, work2, work3, work4,

                pivotval, pivotidx, iipiv, iinfo, optim_mem);

            if(istat != rocblas_status_success)
            {
                return (istat);
            }
        }
    }

    return (rocblas_status_success);
}

template <typename TA, typename TB, typename TX, typename Tc, typename...>
struct gesv_mxp_lu_call
{
    rocblas_status operator()(rocblas_handle handle,
                              const rocblas_int n,
                              const rocblas_int nrhs,
                              void* A,
                              const rocblas_int lda,
                              rocblas_int* ipiv,
                              void* B,
                              const rocblas_int ldb,
                              void* X,
                              const rocblas_int ldx,
                              const rocblas_int max_iter,
                              const double tol,
                              rocblas_int* niter,
                              rocblas_int* info)
    {
        if constexpr(gesv_ex_mxp_lu_accepts<TA, TB, TX, Tc>)
        {
            bool use_pivot = false;
            // no batched/strided support yet
            rocblas_int batch_count = 1;
            rocblas_stride shiftA = 0, strideA = 1;
            rocblas_stride shiftB = 0, strideB = 1;
            rocblas_stride strideP = 1;
            rocblas_stride shiftX = 0, strideX = 1;
            // here we can choose a reduced precision type based on Tc if we need to, otherwise we default to Tc for both
            // return rocsolver_gesv_ex_mxp_lu<TA, Tc, gesv_ex_mxp_lu_reduced_precision_t<Tc>>(...)
            return rocsolver_gesv_ex_mxp_lu<TA, Tc>(handle, n, nrhs, (TA*)A, shiftA, lda, strideA,
                                                    ipiv, strideP, (TB*)B, shiftB, ldb, strideB,
                                                    (TX*)X, shiftX, ldx, strideX, max_iter, tol,
                                                    niter, info, batch_count, use_pivot);
        }
        return rocblas_status_not_implemented;
    }
};

rocblas_status rocsolver_gesv_ex_impl(rocblas_handle handle,
                                      const rocblas_int n,
                                      const rocblas_int nrhs,
                                      void* A,
                                      const rocblas_datatype A_type,
                                      const rocblas_int lda,
                                      rocblas_int* ipiv,
                                      void* B,
                                      const rocblas_datatype B_type,
                                      const rocblas_int ldb,
                                      void* X,
                                      const rocblas_datatype X_type,
                                      const rocblas_int ldx,
                                      const rocblas_int max_iter,
                                      const double tol,
                                      rocblas_int* niter,
                                      rocblas_datatype compute_type,
                                      rocblas_int* info)
{
    using T = void*;
    ROCSOLVER_ENTER_TOP("gesv_ex", "-n", n, "--nrhs", nrhs, "--lda", lda, "--ldb", ldb);

    rocblas_status ret;

    // the most specific cases are tried first

    ret = rocsolver_ex_datatype_dispatch<gesv_homogenous_call>(A_type, B_type, X_type, compute_type,
                                                               handle, n, nrhs, A, lda, ipiv, B, ldb,
                                                               X, ldx, max_iter, tol, niter, info);

    if(ret != rocblas_status_not_implemented)
    {
        return ret;
    }

    ret = rocsolver_ex_datatype_dispatch<gesv_mxp_lu_call>(A_type, B_type, X_type, compute_type,
                                                           handle, n, nrhs, A, lda, ipiv, B, ldb, X,
                                                           ldx, max_iter, tol, niter, info);

    return ret;
}

#undef CHECK_MEM
ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocblas_status rocsolver_gesv_ex(rocblas_handle handle,
                                            const rocblas_int n,
                                            const rocblas_int nrhs,
                                            void* A,
                                            const rocblas_datatype A_type,
                                            const rocblas_int lda,
                                            rocblas_int* ipiv,
                                            void* B,
                                            const rocblas_datatype B_type,
                                            const rocblas_int ldb,
                                            void* X,
                                            const rocblas_datatype X_type,
                                            const rocblas_int ldx,
                                            const rocblas_int max_iter,
                                            const double tol,
                                            rocblas_int* niter,
                                            rocblas_datatype compute_type,
                                            rocblas_int* info)
{
    return rocsolver::rocsolver_gesv_ex_impl(handle, n, nrhs, A, A_type, lda, ipiv, B, B_type, ldb, X,
                                             X_type, ldx, max_iter, tol, niter, compute_type, info);
}
