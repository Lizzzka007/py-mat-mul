#include "matrix.h"
#include "stdio.h"

#define threads_y 32
#define threads_x 16

// TODO: improve computations speed
template <typename T>
void __global__ mat_mul(const T* A, const T* B, T* C,
                const int A_nrow, const int A_ncol, const int B_ncol)
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y; 
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < A_nrow && j < B_ncol)
    {
        T tmp_sum = T(0.0);

        for (int k = 0; k < A_ncol; k++) 
        {
            tmp_sum += A[i * A_ncol + k] * B[k * B_ncol + j];
        }
        C[i * B_ncol + j] = tmp_sum;
    }
}

dim3 get_cuda_grid(const int dimx_n, const int dimy_n)
{
    dim3 grid_blocks(1, 1); 
    grid_blocks.x = dimx_n % threads_y == 0? dimx_n / threads_y : dimx_n / threads_y + 1;
    grid_blocks.y = dimy_n % threads_x == 0? dimy_n / threads_x : dimy_n / threads_x + 1;

    return grid_blocks;
}

template <typename T>
matrix<T, MemType::GPU> matrix<T, MemType::GPU>::operator *(const matrix<T, MemType::GPU>& other) const
{
    std::tuple<int, int> dims = other.get_dims();
    const int A_nrow = nrow;
    const int A_ncol = ncol;
    const int B_ncol = std::get<1>(dims);

    matrix<T, MemType::GPU> Res(A_nrow, B_ncol);

    const T* A = ptr;
    const T* B = other.get_ptr();
    T* C = Res.get_ptr();

    dim3 block_threads(threads_y, threads_x);
    dim3 grid_blocks = get_cuda_grid(nrow, ncol);
    mat_mul<<<grid_blocks, block_threads>>> (A, B, C,
        A_nrow, A_ncol, B_ncol);
    
    return Res;
}

template class matrix<float, MemType::GPU>;
template class matrix<double, MemType::GPU>;
template class matrix<int, MemType::GPU>;