cmake_minimum_required(VERSION 3.19)
project(MatrixPython)

option(INCLUDE_CUDA "GPU build in mode" OFF)

# find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
# set( Python_FIND_UNVERSIONED_NAMES FIRST )
# set(PYTHON_VERSION "3.8") 
# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
# set(PYBIND11_FINDPYTHON ON)
# set(PYBIND11_PYTHON_VERSION=3.8)
# find_package(Python COMPONENTS Interpreter Development REQUIRED VERSION 3.8)
find_package(Python 3.8 COMPONENTS Interpreter Development EXACT)
# find_package(Python 3.8 EXACT)
find_package(pybind11 CONFIG REQUIRED)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)

message(STATUS "pybind11_DIR: ${pybind11_DIR}")

if(INCLUDE_CUDA)
    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    add_definitions(-DENABLE_GPU_COMPUTATIONS)
endif(INCLUDE_CUDA)

add_subdirectory(matrix)
include_directories(matrix/include)
include_directories(matrix/memory_processing/include)

set(SOURCES_CXX 
    matrix_lib.cpp
    matrix_py_lib.cpp
    )
set(HEADERS_CXX 
    matrix_lib.h
    matrix_py_lib.h
    )

set(SOURCES ${SOURCES_CU} ${SOURCES_CXX})
set(HEADERS ${HEADERS_CXX})

pybind11_add_module(py_mat_mul ${HEADERS} ${SOURCES})
target_compile_options(py_mat_mul
            PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fPIC  >
    )

if(INCLUDE_CUDA)
    target_compile_options(py_mat_mul
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -fPIC  >
    )
endif(INCLUDE_CUDA)

target_link_libraries(py_mat_mul PUBLIC matrix memproc)

# set_target_properties(py_mat_mul PROPERTIES
#     PREFIX "" 
#     SUFFIX ".so" 
#     OUTPUT_NAME "cxx_module"
# )