cmake_minimum_required(VERSION 3.18)

project(BLRMatrixSolver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)


find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(blr_core STATIC
    src/blr_matrix.cu
    include/blr_matrix.h
)

target_link_libraries(blr_core PUBLIC
    CUDA::cublas
    CUDA::cusolver
    CUDA::cudart
)

if(CMAKE_BUILD_TYPE MATCHES "Release")
    target_compile_options(blr_core PRIVATE -O3)
endif()


add_executable(blr_test
    main_test.cu
)

target_link_libraries(blr_test PRIVATE blr_core)

message(STATUS "Configured BLR Matrix Project")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")