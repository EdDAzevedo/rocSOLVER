/* ************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once

#include "client_util.hpp"
#include "clientcommon.hpp"
#include "lapack_host_reference.hpp"
#include "norm.hpp"
#include "rocsolver.hpp"
#include "rocsolver_arguments.hpp"
#include "rocsolver_test.hpp"

template <typename T>
void csrrf_sumlu_checkBadArgs(rocblas_handle handle,
                              const rocblas_int n,
                              const rocblas_int nnzL,
                              rocblas_int* ptrL,
                              rocblas_int* indL,
                              T valL,
                              const rocblas_int nnzU,
                              rocblas_int* ptrU,
                              rocblas_int* indU,
                              T valU,
                              rocblas_int* ptrT,
                              rocblas_int* indT,
                              T valT)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(nullptr, n, nnzL, ptrL, indL, valL, nnzU, ptrU,
                                                indU, valU, ptrT, indT, valT),
                          rocblas_status_invalid_handle);

    // values
    // N/A

    // sizes (only check batch_count if applicable)
    // N/A

    // pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, nnzL, (rocblas_int*)nullptr, indL, valL,
                                                nnzU, ptrU, indU, valU, ptrT, indT, valT),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, nnzL, ptrL, (rocblas_int*)nullptr, valL,
                                                nnzU, ptrU, indU, valU, ptrT, indT, valT),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, nnzL, ptrL, indL, (T) nullptr, nnzU,
                                                ptrU, indU, valU, ptrT, indT, valT),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, nnzL, ptrL, indL, valL, nnzU,
                                                (rocblas_int*)nullptr, indU, valU, ptrT, indT, valT),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU,
                                                (rocblas_int*)nullptr, valU, ptrT, indT, valT),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU,
                                                (T) nullptr, ptrT, indT, valT),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU,
                                                valU, (rocblas_int*)nullptr, indT, valT),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU,
                                                valU, ptrT, (rocblas_int*)nullptr, valT),
                          rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, nnzL, ptrL, indL, valL, nnzU, ptrU, indU,
                                                valU, ptrT, indT, (T) nullptr),
                          rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, n, ptrL, indL, valL, 0, ptrU,
                                                (rocblas_int*)nullptr, (T) nullptr, ptrT,
                                                (rocblas_int*)nullptr, (T) nullptr),
                          rocblas_status_success);

    // quick return with zero batch_count if applicable
    // N/A
}

template <typename T>
void testing_csrrf_sumlu_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    rocblas_int n = 1;
    rocblas_int nnzL = 1;
    rocblas_int nnzU = 1;

    // memory allocations
    device_strided_batch_vector<rocblas_int> ptrL(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> indL(1, 1, 1, 1);
    device_strided_batch_vector<T> valL(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> ptrU(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> indU(1, 1, 1, 1);
    device_strided_batch_vector<T> valU(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> ptrT(1, 1, 1, 1);
    device_strided_batch_vector<rocblas_int> indT(1, 1, 1, 1);
    device_strided_batch_vector<T> valT(1, 1, 1, 1);
    CHECK_HIP_ERROR(ptrL.memcheck());
    CHECK_HIP_ERROR(indL.memcheck());
    CHECK_HIP_ERROR(valL.memcheck());
    CHECK_HIP_ERROR(ptrU.memcheck());
    CHECK_HIP_ERROR(indU.memcheck());
    CHECK_HIP_ERROR(valU.memcheck());
    CHECK_HIP_ERROR(ptrT.memcheck());
    CHECK_HIP_ERROR(indT.memcheck());
    CHECK_HIP_ERROR(valT.memcheck());

    // check bad arguments
    csrrf_sumlu_checkBadArgs(handle, n, nnzL, ptrL.data(), indL.data(), valL.data(), nnzU,
                             ptrU.data(), indU.data(), valU.data(), ptrT.data(), indT.data(),
                             valT.data());
}

template <bool CPU, bool GPU, typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_sumlu_initData(rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nnzL,
                          Ud& dptrL,
                          Ud& dindL,
                          Td& dvalL,
                          const rocblas_int nnzU,
                          Ud& dptrU,
                          Ud& dindU,
                          Td& dvalU,
                          Uh& hptrL,
                          Uh& hindL,
                          Th& hvalL,
                          Uh& hptrU,
                          Uh& hindU,
                          Th& hvalU,
                          Uh& hptrT,
                          Uh& hindT,
                          Th& hvalT,
                          const fs::path testcase,
                          bool test = true)
{
    bool mat_zero = (nnzU + nnzL - n == 0);

    // if not matrix zero, read data from files
    if(!mat_zero)
    {
        if(CPU)
        {
            std::string file;

            // read-in L
            file = testcase / "ptrL";
            read_matrix(file, 1, n + 1, hptrL.data(), 1);
            file = testcase / "indL";
            read_matrix(file, 1, nnzL, hindL.data(), 1);
            file = testcase / "valL";
            read_matrix(file, 1, nnzL, hvalL.data(), 1);

            // read-in U
            file = testcase / "ptrU";
            read_matrix(file, 1, n + 1, hptrU.data(), 1);
            file = testcase / "indU";
            read_matrix(file, 1, nnzU, hindU.data(), 1);
            file = testcase / "valU";
            read_matrix(file, 1, nnzU, hvalU.data(), 1);

            // get results (matrix T) if validation is required
            if(test)
            {
                rocblas_int nnzT = nnzL + nnzU - n;
                file = testcase / "ptrT";
                read_matrix(file, 1, n + 1, hptrT.data(), 1);
                file = testcase / "indT";
                read_matrix(file, 1, nnzT, hindT.data(), 1);
                file = testcase / "valT";
                read_matrix(file, 1, nnzT, hvalT.data(), 1);
            }
        }

        if(GPU)
        {
            CHECK_HIP_ERROR(dptrL.transfer_from(hptrL));
            CHECK_HIP_ERROR(dindL.transfer_from(hindL));
            CHECK_HIP_ERROR(dvalL.transfer_from(hvalL));
            CHECK_HIP_ERROR(dptrU.transfer_from(hptrU));
            CHECK_HIP_ERROR(dindU.transfer_from(hindU));
            CHECK_HIP_ERROR(dvalU.transfer_from(hvalU));
        }
    }
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_sumlu_getError(rocblas_handle handle,
                          const rocblas_int n,
                          const rocblas_int nnzL,
                          Ud& dptrL,
                          Ud& dindL,
                          Td& dvalL,
                          const rocblas_int nnzU,
                          Ud& dptrU,
                          Ud& dindU,
                          Td& dvalU,
                          Ud& dptrT,
                          Ud& dindT,
                          Td& dvalT,
                          Uh& hptrL,
                          Uh& hindL,
                          Th& hvalL,
                          Uh& hptrU,
                          Uh& hindU,
                          Th& hvalU,
                          Uh& hptrT,
                          Uh& hindT,
                          Th& hvalT,
                          Uh& hptrTres,
                          Uh& hindTres,
                          Th& hvalTres,
                          double* max_err,
                          const fs::path testcase)
{
    // input data initialization
    csrrf_sumlu_initData<true, true, T>(handle, n, nnzL, dptrL, dindL, dvalL, nnzU, dptrU, dindU,
                                        dvalU, hptrL, hindL, hvalL, hptrU, hindU, hvalU, hptrT,
                                        hindT, hvalT, testcase);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_csrrf_sumlu(
        handle, n, nnzL, dptrL.data(), dindL.data(), dvalL.data(), nnzU, dptrU.data(), dindU.data(),
        dvalU.data(), dptrT.data(), dindT.data(), dvalT.data()));

    CHECK_HIP_ERROR(hptrTres.transfer_from(dptrT));
    CHECK_HIP_ERROR(hindTres.transfer_from(dindT));
    CHECK_HIP_ERROR(hvalTres.transfer_from(dvalT));

    double err = 0;
    rocblas_int nnzT = nnzU + nnzL - n;
    bool mat_zero = (nnzT == 0);

    // if not matrix zero, compare computed results with original result
    if(!mat_zero)
    {
        for(rocblas_int i = 0; i <= n; ++i)
            err += (hptrT[0][i] - hptrTres[0][i]);

        err += nnzT - hptrTres[0][n];

        for(rocblas_int i = 0; i < nnzT; ++i)
        {
            err += (hindT[0][i] - hindTres[0][i]);
            err += (hvalT[0][i] - hvalTres[0][i]);
        }
    }
    // otherwise simply check that ptrT = 0
    else
    {
        for(rocblas_int i = 0; i <= n; ++i)
            err += hptrTres[0][i];
    }

    *max_err = err;
}

template <typename T, typename Td, typename Ud, typename Th, typename Uh>
void csrrf_sumlu_getPerfData(rocblas_handle handle,
                             const rocblas_int n,
                             const rocblas_int nnzL,
                             Ud& dptrL,
                             Ud& dindL,
                             Td& dvalL,
                             const rocblas_int nnzU,
                             Ud& dptrU,
                             Ud& dindU,
                             Td& dvalU,
                             Ud& dptrT,
                             Ud& dindT,
                             Td& dvalT,
                             Uh& hptrL,
                             Uh& hindL,
                             Th& hvalL,
                             Uh& hptrU,
                             Uh& hindU,
                             Th& hvalU,
                             Uh& hptrT,
                             Uh& hindT,
                             Th& hvalT,
                             double* gpu_time_used,
                             double* cpu_time_used,
                             const rocblas_int hot_calls,
                             const int profile,
                             const bool profile_kernels,
                             const bool perf,
                             const fs::path testcase)
{
    *cpu_time_used = nan(""); // no timing on cpu-lapack execution

    csrrf_sumlu_initData<true, false, T>(handle, n, nnzL, dptrL, dindL, dvalL, nnzU, dptrU, dindU,
                                         dvalU, hptrL, hindL, hvalL, hptrU, hindU, hvalU, hptrT,
                                         hindT, hvalT, testcase, false);

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        csrrf_sumlu_initData<false, true, T>(handle, n, nnzL, dptrL, dindL, dvalL, nnzU, dptrU,
                                             dindU, dvalU, hptrL, hindL, hvalL, hptrU, hindU, hvalU,
                                             hptrT, hindT, hvalT, testcase, false);

        CHECK_ROCBLAS_ERROR(rocsolver_csrrf_sumlu(
            handle, n, nnzL, dptrL.data(), dindL.data(), dvalL.data(), nnzU, dptrU.data(),
            dindU.data(), dvalU.data(), dptrT.data(), dindT.data(), dvalT.data()));
    }

    // gpu-lapack performance
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    double start;

    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        csrrf_sumlu_initData<false, true, T>(handle, n, nnzL, dptrL, dindL, dvalL, nnzU, dptrU,
                                             dindU, dvalU, hptrL, hindL, hvalL, hptrU, hindU, hvalU,
                                             hptrT, hindT, hvalT, testcase, false);

        start = get_time_us_sync(stream);
        rocsolver_csrrf_sumlu(handle, n, nnzL, dptrL.data(), dindL.data(), dvalL.data(), nnzU,
                              dptrU.data(), dindU.data(), dvalU.data(), dptrT.data(), dindT.data(),
                              dvalT.data());
        *gpu_time_used += get_time_us_sync(stream) - start;
    }
    *gpu_time_used /= hot_calls;
}

template <typename T>
void testing_csrrf_sumlu(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    rocblas_int n = argus.get<rocblas_int>("n");
    rocblas_int nnzL = argus.get<rocblas_int>("nnzL");
    rocblas_int nnzU = argus.get<rocblas_int>("nnzU");
    rocblas_int hot_calls = argus.iters;

    // check non-supported values
    // N/A

    // check invalid sizes
    bool invalid_size = (n < 0 || nnzL < n || nnzU < 0);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, nnzL, (rocblas_int*)nullptr,
                                                    (rocblas_int*)nullptr, (T*)nullptr, nnzU,
                                                    (rocblas_int*)nullptr, (rocblas_int*)nullptr,
                                                    (T*)nullptr, (rocblas_int*)nullptr,
                                                    (rocblas_int*)nullptr, (T*)nullptr),
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // determine existing test case
    rocblas_int nnzA = nnzL + nnzU - n;
    bool mat_zero = (nnzA == 0);
    if(!mat_zero)
    {
        if(n > 0)
        {
            if(n <= 35)
                n = 20;
            else if(n <= 75)
                n = 50;
            else if(n <= 175)
                n = 100;
            else
                n = 250;
        }

        if(n <= 50) // small case
        {
            if(nnzA <= 80)
                nnzA = 60;
            else if(nnzA <= 120)
                nnzA = 100;
            else
                nnzA = 140;
        }
        else // large case
        {
            if(nnzA <= 400)
                nnzA = 300;
            else if(nnzA <= 600)
                nnzA = 500;
            else
                nnzA = 700;
        }
    }

    // read/set actual nnzL, nnzU and nnzT
    fs::path testcase;
    if(!mat_zero && n > 0)
    {
        testcase = get_sparse_data_dir() / fmt::format("mat_{}_{}", n, nnzA);
        fs::path file = testcase / "ptrL";
        read_last(file, &nnzL);
        file = testcase / "ptrU";
        read_last(file, &nnzU);
    }
    rocblas_int nnzT = nnzL + nnzU - n;

    // memory size query if necessary
    if(argus.mem_query || !USE_ROCBLAS_REALLOC_ON_DEMAND)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_csrrf_sumlu(
            handle, n, nnzL, (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr, nnzU,
            (rocblas_int*)nullptr, (rocblas_int*)nullptr, (T*)nullptr, (rocblas_int*)nullptr,
            (rocblas_int*)nullptr, (T*)nullptr));

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));
        if(argus.mem_query)
        {
            rocsolver_bench_inform(inform_mem_query, size);
            return;
        }

        CHECK_ROCBLAS_ERROR(rocblas_set_device_memory_size(handle, size));
    }

    // determine sizes
    size_t size_ptrL = size_t(n) + 1;
    size_t size_indL = size_t(nnzL);
    size_t size_valL = size_t(nnzL);
    size_t size_ptrU = size_t(n) + 1;
    size_t size_indU = size_t(nnzU);
    size_t size_valU = size_t(nnzU);
    size_t size_ptrT = size_t(n) + 1;
    size_t size_indT = size_t(nnzT);
    size_t size_valT = size_t(nnzT);

    size_t size_ptrTres = 0;
    size_t size_indTres = 0;
    size_t size_valTres = 0;
    if(argus.unit_check || argus.norm_check)
    {
        size_ptrTres = size_ptrT;
        size_indTres = size_indT;
        size_valTres = size_valT;
    }

    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

    // memory allocations
    host_strided_batch_vector<rocblas_int> hptrL(size_ptrL, 1, size_ptrL, 1);
    host_strided_batch_vector<rocblas_int> hindL(size_indL, 1, size_indL, 1);
    host_strided_batch_vector<T> hvalL(size_valL, 1, size_valL, 1);
    host_strided_batch_vector<rocblas_int> hptrU(size_ptrU, 1, size_ptrU, 1);
    host_strided_batch_vector<rocblas_int> hindU(size_indU, 1, size_indU, 1);
    host_strided_batch_vector<T> hvalU(size_valU, 1, size_valU, 1);
    host_strided_batch_vector<rocblas_int> hptrT(size_ptrT, 1, size_ptrT, 1);
    host_strided_batch_vector<rocblas_int> hindT(size_indT, 1, size_indT, 1);
    host_strided_batch_vector<T> hvalT(size_valT, 1, size_valT, 1);
    host_strided_batch_vector<rocblas_int> hptrTres(size_ptrTres, 1, size_ptrTres, 1);
    host_strided_batch_vector<rocblas_int> hindTres(size_indTres, 1, size_indTres, 1);
    host_strided_batch_vector<T> hvalTres(size_valTres, 1, size_valTres, 1);

    device_strided_batch_vector<rocblas_int> dptrL(size_ptrL, 1, size_ptrL, 1);
    device_strided_batch_vector<rocblas_int> dindL(size_indL, 1, size_indL, 1);
    device_strided_batch_vector<T> dvalL(size_valL, 1, size_valL, 1);
    device_strided_batch_vector<rocblas_int> dptrU(size_ptrU, 1, size_ptrU, 1);
    device_strided_batch_vector<rocblas_int> dindU(size_indU, 1, size_indU, 1);
    device_strided_batch_vector<T> dvalU(size_valU, 1, size_valU, 1);
    device_strided_batch_vector<rocblas_int> dptrT(size_ptrT, 1, size_ptrT, 1);
    device_strided_batch_vector<rocblas_int> dindT(size_indT, 1, size_indT, 1);
    device_strided_batch_vector<T> dvalT(size_valT, 1, size_valT, 1);
    CHECK_HIP_ERROR(dptrL.memcheck());
    CHECK_HIP_ERROR(dptrU.memcheck());
    CHECK_HIP_ERROR(dptrT.memcheck());
    if(size_indL)
        CHECK_HIP_ERROR(dindL.memcheck());
    if(size_valL)
        CHECK_HIP_ERROR(dvalL.memcheck());
    if(size_indU)
        CHECK_HIP_ERROR(dindU.memcheck());
    if(size_valU)
        CHECK_HIP_ERROR(dvalU.memcheck());
    if(size_valT)
        CHECK_HIP_ERROR(dvalT.memcheck());
    if(size_indT)
        CHECK_HIP_ERROR(dindT.memcheck());

    // check quick return
    if(n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_csrrf_sumlu(handle, n, nnzL, dptrL.data(), dindL.data(),
                                                    dvalL.data(), nnzU, dptrU.data(), dindU.data(),
                                                    dvalU.data(), dptrT.data(), dindT.data(),
                                                    dvalT.data()),
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
        csrrf_sumlu_getError<T>(handle, n, nnzL, dptrL, dindL, dvalL, nnzU, dptrU, dindU, dvalU,
                                dptrT, dindT, dvalT, hptrL, hindL, hvalL, hptrU, hindU, hvalU, hptrT,
                                hindT, hvalT, hptrTres, hindTres, hvalTres, &max_error, testcase);

    // collect performance data
    if(argus.timing)
        csrrf_sumlu_getPerfData<T>(handle, n, nnzL, dptrL, dindL, dvalL, nnzU, dptrU, dindU, dvalU,
                                   dptrT, dindT, dvalT, hptrL, hindL, hvalL, hptrU, hindU, hvalU,
                                   hptrT, hindT, hvalT, &gpu_time_used, &cpu_time_used, hot_calls,
                                   argus.profile, argus.profile_kernels, argus.perf, testcase);

    // validate results for rocsolver-test
    // using machine precision for tolerance
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 1);

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("n", "nnzL", "nnzU");
            rocsolver_bench_output(n, nnzL, nnzU);

            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_CSRRF_SUMLU(...) \
    extern template void testing_csrrf_sumlu<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_CSRRF_SUMLU, FOREACH_REAL_TYPE, APPLY_STAMP)
