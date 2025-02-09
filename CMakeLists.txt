cmake_minimum_required(VERSION 3.30)
project(abacus CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(FINE_CXX_FLAGS "-Wall -Wextra -Wsuggest-attribute=const -fno-strict-aliasing -Wno-sign-compare")
set(FINE_CXX_FLAGS "${FINE_CXX_FLAGS} -Wno-unknown-pragmas -Wnull-dereference -Wno-switch -Wfloat-equal")
set(FINE_CXX_FLAGS "${FINE_CXX_FLAGS} -Wduplicated-branches -Wformat=2 -Wno-unused-but-set-parameter")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FINE_CXX_FLAGS}")

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all -Xcudafe --display_error_number")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"${FINE_CXX_FLAGS}\"")

include(CheckCompilerFlag)
check_compiler_flag(CUDA -t4 NVCC_THREADS)

find_package(CUDAToolkit REQUIRED)
add_executable(abacus main.cu)

set_target_properties(abacus PROPERTIES
        POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON)

string(SUBSTRING "${CMAKE_CUDA_ARCHITECTURES_NATIVE}" 0 2 COMPUTE_CAPABILITY) # xx-real -> xx

set(ENV{CPM_USE_LOCAL_PACKAGES} ON)
set(ENV{CPM_SOURCE_CACHE} "./cmake/cache")
include(cmake/CPM.cmake)

set(CCCL_ENABLE_UNSTABLE ON)
CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY nvidia/cccl
        FORCE 1
        GIT_TAG main # Fetches the latest commit on the main branch
)
if(CCCL_ADDED)
    target_link_libraries(abacus PRIVATE CCCL::CCCL)
endif()

#CUTLASS business
CPMAddPackage(
        NAME CUTLASS
        GITHUB_REPOSITORY nvidia/cutlass
        GIT_TAG main
        DOWNLOAD_ONLY TRUE
        OPTIONS
        "CUTLASS_NVCC_ARCHS=${COMPUTE_CAPABILITY}"
)
if(CUTLASS_ADDED)
    # header-only
    target_include_directories(abacus SYSTEM PRIVATE "${CUTLASS_SOURCE_DIR}/include")
    set(cublasdx_CUTLASS_ROOT "${CUTLASS_SOURCE_DIR}")
endif ()

CPMAddPackage(
        NAME FMT
        GITHUB_REPOSITORY fmtlib/fmt
        GIT_TAG 11.0.2
        DOWNLOAD_ONLY
)
if(FMT_ADDED)
    target_link_libraries(abacus PRIVATE fmt::fmt)
endif ()

set(MATHDX_VER 25.01)
CPMFindPackage(
        NAME mathdx
        VERSION "${MATHDX_VER}"
        FIND_PACKAGE_ARGUMENTS "REQUIRED COMPONENTS cublasdx CONFIG"
)

target_link_libraries(abacus PRIVATE mathdx::cublasdx)
target_link_libraries(abacus PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::nvml CUDA::nvtx3 CUDA::cublas)

target_compile_options(abacus PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xfatbin -compress-all>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-t0; --generate-line-info>
        $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(abacus PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Og;-g;>
            $<$<COMPILE_LANGUAGE:CUDA>:-O1; -g; -G>
    )
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(abacus PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_${COMPUTE_CAPABILITY},code=lto_${COMPUTE_CAPABILITY}>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -v;--expt-relaxed-constexpr>
    )
endif ()