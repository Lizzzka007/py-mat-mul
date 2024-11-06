project(Matrix)
cmake_minimum_required(VERSION 3.19)

option(INCLUDE_CUDA "GPU build in mode" OFF)
option(INCLUDE_OPENMP "OpenMP build in mode" OFF)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)

if(INCLUDE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DENABLE_OPENMP_COMPUTATIONS)
endif(INCLUDE_OPENMP)

if(INCLUDE_CUDA)
    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    add_definitions(-DENABLE_GPU_COMPUTATIONS)
    add_definitions(-DINCLUDE_CUDA)
endif(INCLUDE_CUDA)

include(FetchContent)
    FetchContent_Declare(memory_faucet                                   
    GIT_REPOSITORY http://tesla.parallel.ru/Lizzzka007/memory-faucet.git                       
    GIT_TAG origin/main
    )
FetchContent_MakeAvailable(memory_faucet)

FetchContent_Declare(
        memory_processing
        GIT_REPOSITORY http://tesla.parallel.ru/Lizzzka007/memory_processing.git
        GIT_TAG        main
)
FetchContent_MakeAvailable(memory_processing)


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

target_link_libraries(matrix PUBLIC memory-faucet memproc)
if(INCLUDE_OPENMP)
    target_link_libraries(matrix PRIVATE OpenMP::OpenMP_CXX)
endif(INCLUDE_OPENMP)

target_include_directories(matrix PUBLIC ${memory_faucet_SOURCE_DIR}/include ${memory_processing_SOURCE_DIR}/include)