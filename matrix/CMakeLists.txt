project(Matrix)
cmake_minimum_required(VERSION 3.19)

option(INCLUDE_CUDA "GPU build in mode" OFF)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)

if(INCLUDE_CUDA)
    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    add_definitions(-DENABLE_GPU_COMPUTATIONS)
endif(INCLUDE_CUDA)

add_subdirectory(memory_processing)
include_directories(memory_processing/include)
include_directories(include)

set(SOURCES_CXX 
    src/matrix.cpp
    )
set(HEADERS_CXX 
    include/matrix.h
    )
if(INCLUDE_CUDA)
    set(SOURCES_CU 
    src/matrix.cu
    )
endif(INCLUDE_CUDA)

set(SOURCES ${SOURCES_CU} ${SOURCES_CXX})
set(HEADERS ${HEADERS_CXX})

add_library(matrix STATIC ${HEADERS} ${SOURCES})

target_compile_options(matrix
            PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -fPIC  >
    )
if(INCLUDE_CUDA)
    target_compile_options(matrix
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -fPIC  >
    )
endif(INCLUDE_CUDA)

target_link_libraries(matrix PUBLIC memproc)