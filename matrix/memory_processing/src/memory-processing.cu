#include "memory-processing.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#define threads_x 32
#define threads_y 16

namespace memproc
{
    template<>
    bool dealloc<MemType::GPU>(void *&array, size_t &allocated_size)
    {
        if(allocated_size > 0)
        {
            cudaFree(array);
            allocated_size = 0;
        }

        return true;
    }

    template<>
    bool dealloc<MemType::GPU>(void *&array)
    {
        cudaFree(array);
        return true;
    }

    template <>
    bool alloc<MemType::GPU>(void *&array, const size_t new_size)
    {
        cudaMalloc ( (void **)&array, new_size);
        cudaMemset(array, 0, new_size);

        return true;
    }

    template <>
    bool realloc<MemType::GPU>(void *&array, size_t &allocated_size, const size_t new_size)
    {
        if(new_size > allocated_size)
        {
            if(allocated_size > 0) dealloc<MemType::GPU>(array, allocated_size);
            allocated_size = new_size;
            cudaMalloc ( (void **)&array, new_size);
            cudaMemset(array, 0, new_size);
        }

        return true;
    }

    template <>
    bool memcopy<MemType::GPU, MemType::CPU>(void *dst, const void* src, const size_t copy_elem_size)
    {
        if(copy_elem_size == 0) return true;
        cudaMemcpy ( dst, src, copy_elem_size, cudaMemcpyHostToDevice);

        return true;
    }

    template <>
    bool memcopy<MemType::CPU, MemType::GPU>(void *dst, const void* src, const size_t copy_elem_size)
    {
        if(copy_elem_size == 0) return true;
        cudaMemcpy ( dst, src, copy_elem_size, cudaMemcpyDeviceToHost);
        return true;
    }

    template <>
    bool memcopy<MemType::GPU, MemType::GPU>(void *dst, const void* src, const size_t copy_elem_size)
    {
        if(copy_elem_size == 0 || dst == src) return true;
        cudaMemcpy ( dst, src, copy_elem_size, cudaMemcpyDeviceToDevice);
        return true;
    }

    dim3 get_cuda_grid(const int dimx_n, const int dimy_n)
    {
        dim3 grid_blocks(1, 1); 
        grid_blocks.x = dimx_n % threads_y == 0? dimx_n / threads_y : dimx_n / threads_y + 1;
        grid_blocks.y = dimy_n % threads_x == 0? dimy_n / threads_x : dimy_n / threads_x + 1;

        return grid_blocks;
    }
}