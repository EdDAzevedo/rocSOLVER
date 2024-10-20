#pragma once
// ----------------------------------------------
// determine block sizes for simple_gemm_kernel()
// try to fit submatrices in LDS
// ----------------------------------------------

static int get_num_cu(int deviceId = 0)
{
    int ival = 0;
    auto const attr = hipDeviceAttributeMultiprocessorCount;
    HIP_CHECK(hipDeviceGetAttribute(&ival, attr, deviceId));
    return (ival);
}

template <typename T, typename I>
__host__ __device__ static void get_gemm_nb(char const transA,
                                            char const transB,
                                            I const m,
                                            I const n,
                                            I const k,
                                            I* p_nb_m,
                                            I* p_nb_n,
                                            I* p_nb_k)
{
    assert(p_nb_m != nullptr);
    assert(p_nb_n != nullptr);
    assert(p_nb_k != nullptr);

    I const max_lds = (64 * 1024) / sizeof(T);
    I const nb = (sizeof(T) == 4) ? 64 : (sizeof(T) == 8) ? 48 : 32;
    assert((3 * nb * nb) <= max_lds);

    *p_nb_m = nb;
    *p_nb_n = nb;
    *p_nb_k = nb;

    bool const use_default_nb = true;
    if(use_default_nb)
    {
        return;
    }

    I nb_m = std::min(m, nb);
    I nb_n = std::min(n, nb);
    I nb_k = std::min(k, nb);

    // need to fit
    // nb_m * nb_n + nb_m * nb_k + nb_k * nb_n <= max_lds
    // nb_k * (nb_m + nb_n) <= (max_lds - nb_m * nb_n)
    // nb_k = (max_lds - nb_m * nb_n)/(nb_m + nb_n);

    auto const max_mnk = std::max(m, std::max(n, k));

    // ------------------------------
    // try to increase the block size
    // ------------------------------
    if(m == max_mnk)
    {
        nb_m = (max_lds - nb_k * nb_n) / (nb_n + nb_k);
    }
    else if(n == max_mnk)
    {
        nb_n = (max_lds - nb_m * nb_k) / (nb_m + nb_k);
    }
    else if(k == max_mnk)
    {
        nb_k = (max_lds - nb_m * nb_n) / (nb_m + nb_n);
    }

    if(idebug >= 1)
    {
        printf("get_gemm_nb:m=%d,n=%d,k=%d,  nb_m=%d,nb_n=%d,nb_k=%d \n", m, n, k, nb_m, nb_n, nb_k);
    }

    nb_m = std::max(nb_m, 1);
    nb_n = std::max(nb_n, 1);
    nb_k = std::max(nb_k, 1);

#ifdef NDEBUG
#else

    {
        auto const size_A = nb_m * nb_k;
        auto const size_B = nb_k * nb_n;
        auto const size_C = nb_m * nb_n;

        assert((size_A + size_B + size_C) <= max_lds);
    }
#endif

    *p_nb_m = nb_m;
    *p_nb_n = nb_n;
    *p_nb_k = nb_k;
}

// ------------------------------------------
// scale_beta_kernel to scale a matrix by beta
//
// launch as dim3(nbx,nby,nbz), dim3(nx,ny,1)
// ------------------------------------------

template <typename T, typename I, typename Istride, typename UA>
static __global__ void scale_beta_kernel(I const m,
                                         I const n,
                                         T const beta,
                                         UA Amat,
                                         Istride const shift_Amat,
                                         I const ldA,
                                         Istride const stride_Amat,
                                         I const batch_count)
{
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    if(beta == 1)
    {
        return;
    }

    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;

    I const i_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    I const i_inc = hipGridDim_x * hipBlockDim_x;

    I const j_start = hipThreadIdx_y + hipBlockIdx_y * hipBlockDim_y;
    I const j_inc = hipGridDim_y * hipBlockDim_y;

    bool const is_beta_zero = (beta == 0);
    T const zero = 0;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T* const A_ = load_ptr_batch(Amat, bid, shift_Amat, stride_Amat);

        auto idx2D = [](auto i, auto j, auto ld) { return (i + j * static_cast<int64_t>(ld)); };

        auto A = [=](auto i, auto j) -> T& { return (A_[idx2D(i, j, ldA)]); };

        if(is_beta_zero)
        {
            // -----------
            // assign zero
            // -----------

            for(I j = j_start; j < n; j += j_inc)
            {
                for(I i = i_start; i < m; i += i_inc)
                {
                    A(i, j) = zero;
                }
            }
        }
        else
        {
            // ----------------
            // multiply by beta
            // ----------------

            for(I j = j_start; j < n; j += j_inc)
            {
                for(I i = i_start; i < m; i += i_inc)
                {
                    A(i, j) *= beta;
                }
            }
        }
    } // end for bid
}

template <typename T, typename I, typename Istride, typename UC>
static void scale_beta_template(rocblas_handle handle,

                                I const m,
                                I const n,
                                T const beta,

                                UC Cmat,
                                Istride const shift_Cmat,
                                I const ldC,
                                Istride const stride_Cmat,

                                I const batch_count)
{
    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return;
    }

    if(beta == 1)
    {
        return;
    }

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto const nx = 32;
    auto const ny = 32;

    auto const max_blocks = 64 * 1000;
    auto const nbx = std::min(max_blocks, ceil(m, nx));
    auto const nby = std::min(max_blocks, ceil(n, ny));
    auto const nbz = std::min(max_blocks, batch_count);

    ROCSOLVER_LAUNCH_KERNEL((scale_beta_kernel<T, I, Istride, UC>), dim3(nbx, nby, nbz),
                            dim3(nx, ny, 1), 0, stream,

                            m, n, beta,

                            Cmat, shift_Cmat, ldC, stride_Cmat,

                            batch_count);
}

// ------------------------------------------
// simple_gemm_kernel
//
// launch as dim3(nbx,nby,nbz), dim3(nx,ny,1)
// ------------------------------------------

template <typename T, typename I, typename Istride, typename UA, typename UB, typename UC>
static __global__ void simple_gemm_kernel(

    char const transA,
    char const transB,

    I const m,
    I const n,
    I const k,

    T const alpha,

    UA Amat,
    Istride const shift_Amat,
    I const ldA,
    Istride const stride_Amat,

    UB Bmat,
    Istride const shift_Bmat,
    I const ldB,
    Istride const stride_Bmat,

    // note no beta

    UC Cmat,
    Istride const shift_Cmat,
    I const ldC,
    Istride const stride_Cmat,

    I const batch_count)
{
    using S = decltype(std::real(T{}));
    bool constexpr is_complex = rocblas_is_complex<T>;

    bool const has_work = (m >= 1) && (n >= 1) && (k >= 1);
    if(!has_work)
    {
        return;
    };

    bool const is_transpose_A = (transA == 'T') || (transA == 't');
    bool const is_conj_transpose_A = (transA == 'C') || (transA == 'c');
    bool const is_no_transpose_A = (!is_transpose_A) && (!is_conj_transpose_A);

    bool const is_transpose_B = (transB == 'T') || (transB == 't');
    bool const is_conj_transpose_B = (transB == 'C') || (transB == 'c');
    bool const is_no_transpose_B = (!is_transpose_B) && (!is_conj_transpose_B);

    I const nbx = hipGridDim_x;
    I const nby = hipGridDim_y;
    I const nbz = hipGridDim_z;

    I const ib_inc = nbx;
    I const ib_start = hipBlockIdx_x;

    I const jb_inc = nby;
    I const jb_start = hipBlockIdx_y;

    // ---------------------------------------------
    // use only 1 thread block in k-dimension
    // for simplicity
    // ---------------------------------------------
    I bid_inc = nbz;
    I bid_start = hipBlockIdx_z;
    I kb_start = 0;
    I kb_inc = 1;

    I const nx = hipBlockDim_x;
    I const ny = hipBlockDim_y;

    I const i_inc = nx;
    I const j_inc = ny;

    I const i_start = hipThreadIdx_x;
    I const j_start = hipThreadIdx_y;

    I nb = (sizeof(T) == 4) ? 64 : (sizeof(T) == 8) ? 48 : 32;
    I nb_m = nb;
    I nb_n = nb;
    I nb_k = nb;
    get_gemm_nb<T>(transA, transB, m, n, k, &nb_m, &nb_n, &nb_k);

    auto ceil = [](auto n, auto nb) { return ((n - 1) / nb + 1); };

    auto idx2F
        = [](auto i, auto j, auto ld) { return ((i - 1) + (j - 1) * static_cast<int64_t>(ld)); };

    auto merge = [](auto lcond, auto t_value, auto f_value) { return ((lcond) ? t_value : f_value); };

    size_t const max_lds = 64 * 1024;
    size_t const lmem_size = max_lds / sizeof(T);
    __shared__ T lmem[lmem_size];

    I const mblocks = ceil(m, nb_m);
    I const nblocks = ceil(n, nb_n);
    I const kblocks = ceil(k, nb_k);

    if(batch_count == 1)
    {
        kb_start = hipBlockIdx_z;
        kb_inc = nbz;
        bid_start = 0;
        bid_inc = 1;
    }
    else if(kblocks == 1)
    {
        kb_start = 0;
        kb_inc = 1;
        bid_start = hipBlockIdx_z;
        bid_inc = nbz;
    }
    else
    {
        // ----------------
        // split nbz blocks
        // ----------------

        I const nbz_sqrt = std::sqrt(nbz);
        I ni = std::min(nbz_sqrt, std::min(kblocks, batch_count));
        while((nbz % ni) != 0)
        {
            ni--;
        }
        auto const nj = nbz / ni;
        assert(ni >= 1);
        assert(nj >= 1);
        assert((ni * nj) == nbz);

        auto const min_ni_nj = std::min(ni, nj);
        auto const max_ni_nj = std::max(ni, nj);

        kb_inc = (kblocks < batch_count) ? min_ni_nj : max_ni_nj;
        bid_inc = (kblocks < batch_count) ? max_ni_nj : min_ni_nj;

        // ---------------------------------------------
        // partition nbz blocks as  kb_inc by bid_inc
        // hipBlockIdx_z = kb_start + bid_start * kb_inc
        // ---------------------------------------------
        kb_start = (hipBlockIdx_z % kb_inc);
        bid_start = (hipBlockIdx_z - kb_start) / kb_inc;
    }

    T* pfree = (T*)&(lmem[0]);

    auto const size_Csh = nb_m * nb_n;
    T* const Csh_ = pfree;
    pfree += size_Csh;

    auto const size_Ash = nb_m * nb_k;
    T* const Ash_ = pfree;
    pfree += size_Ash;

    auto const size_Bsh = nb_k * nb_n;
    T* const Bsh_ = pfree;
    pfree += size_Bsh;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        T const* const A_ = load_ptr_batch(Amat, bid, shift_Amat, stride_Amat);
        T const* const B_ = load_ptr_batch(Bmat, bid, shift_Bmat, stride_Bmat);
        T* const C_ = load_ptr_batch(Cmat, bid, shift_Cmat, stride_Cmat);

        auto A = [=](auto i, auto j) -> const T& { return (A_[idx2F(i, j, ldA)]); };
        auto B = [=](auto i, auto j) -> const T& { return (B_[idx2F(i, j, ldB)]); };
        auto C = [=](auto i, auto j) -> T& { return (C_[idx2F(i, j, ldC)]); };

        for(I jb = 1 + jb_start; jb <= nblocks; jb += jb_inc)
        {
            for(I ib = 1 + ib_start; ib <= mblocks; ib += ib_inc)
            {
                auto const ic_start = 1 + (ib - 1) * nb_m;
                auto const ic_end = std::min(m, ic_start + nb_m - 1);

                auto const jc_start = 1 + (jb - 1) * nb_n;
                auto const jc_end = std::min(n, jc_start + nb_n - 1);

                auto const ic1 = ic_start;
                auto const ic2 = ic_end;

                auto const jc1 = jc_start;
                auto const jc2 = jc_end;

                auto const nrows_C = (ic2 - ic1 + 1);
                auto const ncols_C = (jc2 - jc1 + 1);

                auto Csh = [=](auto i, auto j) -> T& {
                    assert((1 <= i) && (i <= nrows_C));
                    assert((1 <= j) && (j <= ncols_C));
                    return (Csh_[(i - 1) + (j - 1) * nrows_C]);
                };

                assert((nrows_C * ncols_C) <= size_Csh);

                // --------------------------------
                // Csh( 1:nrows_C, 1:ncols_C ) = 0;
                // --------------------------------

                __syncthreads();

                for(auto j = 1 + j_start; j <= ncols_C; j += j_inc)
                {
                    for(auto i = 1 + i_start; i <= nrows_C; i += i_inc)
                    {
                        Csh(i, j) = 0;
                    }
                }

                __syncthreads();

                for(auto kb = 1 + kb_start; kb <= kblocks; kb += kb_inc)
                {
                    auto const ik_start = 1 + (kb - 1) * nb_k;
                    auto const ik_end = std::min(k, ik_start + nb_k - 1);

                    auto const ik1 = ik_start;
                    auto const ik2 = ik_end;

                    // -----------------------------------------------------------------
                    // C(ic1:ic2, jc1:jc2) <- op(A(ia1:ia2, ja1:ja2)) * op(B( ib1:ib2,
                    // jb1:jb2))
                    // -----------------------------------------------------------------

                    auto const ia1 = merge(is_no_transpose_A, ic1, ik1);
                    auto const ia2 = merge(is_no_transpose_A, ic2, ik2);

                    auto const ja1 = merge(is_no_transpose_A, ik1, ic1);
                    auto const ja2 = merge(is_no_transpose_A, ik2, ic2);

                    auto const ib1 = merge(is_no_transpose_B, ik1, jc1);
                    auto const ib2 = merge(is_no_transpose_B, ik2, jc2);

                    auto const jb1 = merge(is_no_transpose_B, jc1, ik1);
                    auto const jb2 = merge(is_no_transpose_B, jc2, ik2);

                    auto const nrows_A = (ia2 - ia1 + 1);
                    auto const ncols_A = (ja2 - ja1 + 1);

                    // ---------------------------------------------------
                    // Ash( 1:nrows_A, 1:ncols_A ) = A( ia1:ia2, ja1:ja2 );
                    // ---------------------------------------------------

                    auto Ash = [=](auto i, auto j) -> T& {
                        assert((1 <= i) && (i <= nrows_A));
                        assert((1 <= j) && (j <= ncols_A));
                        return (Ash_[(i - 1) + (j - 1) * nrows_A]);
                    };

                    assert((nrows_A * ncols_A) <= size_Ash);

                    __syncthreads();

                    for(auto j = 1 + j_start; j <= ncols_A; j += j_inc)
                    {
                        for(auto i = 1 + i_start; i <= nrows_A; i += i_inc)
                        {
                            auto const ia = (ia1 - 1) + i;
                            auto const ja = (ja1 - 1) + j;
                            Ash(i, j) = A(ia, ja);
                        }
                    }
                    __syncthreads();

                    auto const nrows_B = (ib2 - ib1 + 1);
                    auto const ncols_B = (jb2 - jb1 + 1);

                    // -----------------------------------------------
                    // Bsh(1:nrows_B, 1:ncols_B) = B(ib1:ib2, jb1:jb2);
                    // -----------------------------------------------
                    auto Bsh = [=](auto i, auto j) -> T& {
                        assert((1 <= i) && (i <= nrows_B));
                        assert((1 <= j) && (j <= ncols_B));
                        return (Bsh_[(i - 1) + (j - 1) * nrows_B]);
                    };

                    assert((nrows_B * ncols_B) <= size_Bsh);

                    __syncthreads();

                    for(auto j = 1 + j_start; j <= ncols_B; j += j_inc)
                    {
                        for(auto i = 1 + i_start; i <= nrows_B; i += i_inc)
                        {
                            auto const ib = (ib1 - 1) + i;
                            auto const jb = (jb1 - 1) + j;
                            Bsh(i, j) = B(ib, jb);
                        }
                    }
                    __syncthreads();

                    for(auto j = 1 + j_start; j <= ncols_C; j += j_inc)
                    {
                        for(auto i = 1 + i_start; i <= nrows_C; i += i_inc)
                        {
                            T cij = 0;
                            auto const nk = merge(is_no_transpose_A, ncols_A, nrows_A);

                            bool constexpr use_pointers = true;
                            if(use_pointers)
                            {
                                I const kk = 1;
                                T const* __restrict__ ap
                                    = (is_no_transpose_A) ? &(Ash(i, kk)) : &(Ash(kk, i));
                                I ap_inc = (is_no_transpose_A) ? nrows_A : 1;

                                T const* __restrict__ bp
                                    = (is_no_transpose_B) ? &(Bsh(kk, j)) : &(Bsh(j, kk));
                                I const bp_inc = (is_no_transpose_B) ? 1 : nrows_B;
                                for(I kk = 1; kk <= nk; kk++)
                                {
                                    T const aik = (is_conj_transpose_A) ? conj(*ap) : *ap;
                                    T const bkj = (is_conj_transpose_B) ? conj(*bp) : *bp;

                                    cij += aik * bkj;

                                    ap += ap_inc;
                                    bp += bp_inc;
                                }
                            }
                            else
                            {
                                for(I kk = 1; kk <= nk; kk++)
                                {
                                    T const aik = (is_no_transpose_A) ? Ash(i, kk)
                                        : (is_transpose_A)            ? Ash(kk, i)
                                                                      : conj(Ash(kk, i));

                                    T const bkj = (is_no_transpose_B) ? Bsh(kk, j)
                                        : (is_transpose_B)            ? Bsh(j, kk)
                                                                      : conj(Bsh(j, kk));

                                    cij += aik * bkj;
                                } // end for kk
                            }

                            Csh(i, j) += cij;
                        } // end for i
                    } // end for j

                    __syncthreads();
                } // for kb

                // -----------------------------------------------------------
                // C(ic1:ic2, jc1:jc2) +=  alpha * Csh( 1:nrows_C, 1:ncols_C );
                // -----------------------------------------------------------

                auto gatomicAdd = [](T* p, T value) {
                    S* px = (S*)p;
                    if constexpr(is_complex)
                    {
                        atomicAdd(px, std::real(value));
                        atomicAdd(px + 1, std::imag(value));
                    }
                    else
                    {
                        atomicAdd(px, value);
                    }
                };

                __syncthreads();
                for(auto j = 1 + j_start; j <= ncols_C; j += j_inc)
                {
                    for(auto i = 1 + i_start; i <= nrows_C; i += i_inc)
                    {
                        auto const ic = (ic1 - 1) + i;
                        auto const jc = (jc1 - 1) + j;

                        if(kb_inc == 1)
                        {
                            C(ic, jc) += alpha * Csh(i, j);
                        }
                        else
                        {
                            gatomicAdd(&(C(ic, jc)), (alpha * Csh(i, j)));
                        }
                    }
                }
                __syncthreads();

            } // for ib
        } // for jb

    } // end for bid
}

template <typename T, typename I, typename Istride, typename UA, typename UB, typename UC>
static rocblas_status roclapack_simple_gemm_template(rocblas_handle handle,
                                                     rocblas_operation const transA,
                                                     rocblas_operation const transB,
                                                     I const m,
                                                     I const n,
                                                     I const k,

                                                     T* const p_alpha,

                                                     UA Amat,
                                                     Istride const shift_Amat,
                                                     I const ldA,
                                                     Istride const stride_Amat,

                                                     UB Bmat,
                                                     Istride const shift_Bmat,
                                                     I const ldB,
                                                     Istride const stride_Bmat,

                                                     T* const p_beta,

                                                     UC Cmat,
                                                     Istride const shift_Cmat,
                                                     I const ldC,
                                                     Istride const stride_Cmat,

                                                     I const batch_count,
                                                     void* workArr = nullptr)
{
    auto op2char = [](rocblas_operation transA) -> char {
        char const c_transA = (transA == rocblas_operation_none) ? 'N'
            : (transA == rocblas_operation_transpose)            ? 'T'
            : (transA == rocblas_operation_conjugate_transpose)  ? 'C'
                                                                 : 'X';
        return (c_transA);
    };

    char const c_transA = op2char(transA);
    char const c_transB = op2char(transB);

#ifdef NDEBUG
#else
    if(idebug >= 1)
    {
        printf("roclapack_simple_gemm:transA=%c,transB=%c,m=%d,n=%d,k=%d \n", c_transA, c_transB, m,
               n, k);
    }
#endif

    bool const has_work = (m >= 1) && (n >= 1) && (batch_count >= 1);
    if(!has_work)
    {
        return (rocblas_status_success);
    }

    assert(p_beta != nullptr);
    assert(p_alpha != nullptr);

    T const beta = *p_beta;
    T const alpha = *p_alpha;

    auto ceil = [](auto n, auto nb) { return (1 + ((n - 1) / nb)); };

    // scale_beta_template<T,I,Istride, UC>(handle, m, n, beta, Cmat, shift_Cmat,
    // ldC, stride_Cmat, batch_count);
    scale_beta_template<T, I, Istride, UC>(handle, m, n, beta, Cmat, shift_Cmat, ldC, stride_Cmat,
                                           batch_count);

    I nb = (sizeof(T) == 4) ? 64 : (sizeof(T) == 8) ? 48 : 32;
    I nb_m = nb;
    I nb_n = nb;
    I nb_k = nb;

    get_gemm_nb<T>(c_transA, c_transB, m, n, k, &nb_m, &nb_n, &nb_k);

    auto const kblocks = ceil(k, nb_k);

    I const max_blocks = 64 * 1000;
    I const max_threads = 1024;

    I const nx = std::max(1, std::min(max_threads, nb_m));
    I const ny = std::max(1, max_threads / nx);

    auto const num_cu = get_num_cu();

    I const nbx = std::min(max_blocks, ceil(m, nx));
    I const nby = std::min(max_blocks, ceil(n, ny));
    I const nbz = std::min(max_blocks, std::min(num_cu, kblocks) * batch_count);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    ROCSOLVER_LAUNCH_KERNEL((simple_gemm_kernel<T, I, Istride, UA, UB, UC>), dim3(nbx, nby, nbz),
                            dim3(nx, ny, 1), 0, stream,

                            c_transA, c_transB, m, n, k, alpha,

                            Amat, shift_Amat, ldA, stride_Amat,

                            Bmat, shift_Bmat, ldB, stride_Bmat,

                            // note no beta

                            Cmat, shift_Cmat, ldC, stride_Cmat, batch_count);

    return (rocblas_status_success);
}
