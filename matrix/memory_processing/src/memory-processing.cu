#include "memory-processing.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

#define threads_x 32
#define threads_y 16

namespace memproc
{
    // namespace memproc_kernel
    // {
    //     void __global__ memcopy(char *dst, const char* src, const size_t elem_stride,  
    //         const int stride_row, const int stride_col,
    //         const int nrow, const int ncol)
    //     {
    //         const int i = blockIdx.y * blockDim.y + threadIdx.y;
    //         const int j = blockIdx.x * blockDim.x + threadIdx.x;

    //         if(i < nrow && j < ncol)
    //         {
    //             *(dst + (i * stride_col + j) * elem_stride) = *(src + (i * ncol + j) * elem_stride);
    //         }
    //     }
    // }

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

    // template <>
    // bool memcopy<MemType::GPU, MemType::GPU>(void *dst, const void* src, const size_t elem_stride, 
    //             const int stride_row, const int stride_col,
    //             const int nrow, const int ncol)
    // {
    //     if(stride_row * stride_col * nrow * ncol * elem_stride == 0) return true;

    //     char* ch_dst = static_cast<char*>(dst); 
    //     const char* ch_src = static_cast<const char*>(src); 

    //     dim3 block_threads(threads_y, threads_x);
    //     dim3 grid_blocks = get_cuda_grid(nrow, ncol);
    //     memproc_kernel::memcopy<<<grid_blocks, block_threads>>> (ch_dst, ch_src, elem_stride,
    //         stride_row, stride_col,
    //         nrow, ncol);
        
    //     return true;
    // }

    // template <>
    // bool memcopy<MemType::CPU, MemType::GPU>(void *dst, const void* src, const size_t elem_stride, 
    //             const int stride_row, const int stride_col,
    //             const int nrow, const int ncol)
    // {
    //     if(stride_row * stride_col * nrow * ncol * elem_stride == 0) return true;

    //     void * src_cpu;
    //     const size_t copy_elem_size = nrow * ncol * elem_stride;
    //     alloc<MemType::CPU>((void *&)src_cpu, copy_elem_size);
    //     memcopy<MemType::CPU, MemType::GPU>(src_cpu, src, copy_elem_size);
    //     memcopy<MemType::CPU, MemType::CPU>(dst, src_cpu, elem_stride, 
    //         stride_row, stride_col,
    //         nrow, ncol);
    //     dealloc<MemType::CPU>(src_cpu);
        
    //     return true;
    // }

    // template <>
    // bool memcopy<MemType::GPU, MemType::CPU>(void *dst, const void* src, const size_t elem_stride, 
    //             const int stride_row, const int stride_col,
    //             const int nrow, const int ncol)
    // {
    //     if(stride_row * stride_col * nrow * ncol * elem_stride == 0) return true;

    //     void * src_cpu;
    //     const size_t copy_elem_size = nrow * ncol * elem_stride;
    //     alloc<MemType::GPU>((void *&)src_cpu, copy_elem_size);
    //     memcopy<MemType::GPU, MemType::CPU>(src_cpu, src, copy_elem_size);
    //     memcopy<MemType::GPU, MemType::GPU>(dst, src_cpu, elem_stride, 
    //         stride_row, stride_col,
    //         nrow, ncol);
    //     dealloc<MemType::GPU>(src_cpu);
        
    //     return true;
    // }
}