cmake_minimum_required(VERSION 3.19)

option(INCLUDE_CUDA "GPU build in mode" OFF)
option(INCLUDE_OPENMP "OpenMP build in mode" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
if(INCLUDE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif(INCLUDE_CUDA)

project(MatrixPython)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

if(INCLUDE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    message(FATAL_ERROR "
                    CMake will not pass any architecture flags to the compiler 
                    because the CUDA architecture is not set. You should specify 
                    an architecture: set -DCMAKE_CUDA_ARCHITECTURES=<N>.")
    endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    add_definitions(-DENABLE_GPU_COMPUTATIONS)
endif(INCLUDE_CUDA)

add_subdirectory(matrix)
include_directories(include)
include_directories(matrix/include)
include_directories(matrix/memory_processing/include)

set(SOURCES_CXX 
    src/matrix_lib.cpp
    src/matrix_py_lib.cpp
    )
set(HEADERS_CXX 
    include/matrix_lib.h
    )

set(SOURCES ${SOURCES_CU} ${SOURCES_CXX})
set(HEADERS ${HEADERS_CXX})

pybind11_add_module(py_mat_mul ${HEADERS} ${SOURCES})

target_link_libraries(py_mat_mul PUBLIC matrix)
set_target_properties(py_mat_mul PROPERTIES
    PREFIX "" 
    SUFFIX ".so" 
    OUTPUT_NAME "py_mat_mul"
)

# Configure tests for pytest ----------------------------------------------------------
add_custom_command(
    TARGET py_mat_mul POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_BINARY_DIR}/tests)

add_custom_command(
    TARGET py_mat_mul POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_BINARY_DIR}/py_mat_mul.so
    ${CMAKE_CURRENT_BINARY_DIR}/tests/)

set(INPUT_TEST_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_config.py.in")
set(OUTPUT_TEST_CONFIG "${CMAKE_CURRENT_BINARY_DIR}/tests/test_config.py")

if(INCLUDE_CUDA)
    set(GPU_COMPILATION_MODE "True")
else(INCLUDE_CUDA)
    set(GPU_COMPILATION_MODE "False")
endif(INCLUDE_CUDA)

add_custom_command(
    OUTPUT ${OUTPUT_TEST_CONFIG}
    COMMAND ${CMAKE_COMMAND} -D GPU_COMPILATION_MODE=${GPU_COMPILATION_MODE} -P ${CMAKE_CURRENT_SOURCE_DIR}/generate_tests_config.cmake
    DEPENDS ${INPUT_TEST_CONFIG}
    COMMENT "Configuring test-config.py"
)

add_custom_target(generate_config ALL DEPENDS ${OUTPUT_TEST_CONFIG})
# ----------------------------------------------------------

enable_testing()
add_test(
  NAME py_mat_mul_pytest
  COMMAND pytest -s 
)
