#include "matrix.h"
#include "MemoryProcessing.h"

#ifdef ENABLE_OPENMP_COMPUTATIONS
#include <omp.h>
#endif

template <typename T, MemType dev>
matrix_base<T, dev>::matrix_base()
{
    allocated_size = 0;
    nrow = 0;
    ncol = 0;
    buf = memBuf<dev>();
}

template <typename T, MemType dev>
template <MemType mem_in>
void matrix_base<T, dev>::set_array(T* array, const int nrow_in, const int ncol_in)
{
    if (mem_in != dev)
    {
        nrow = nrow_in;
        ncol = ncol_in;
        allocated_size = nrow_in * ncol_in * sizeof(T);  
        buf.get_memory(allocated_size);
        ptr = static_cast<T*>(buf.ptr());
        memproc::memcopy<dev, mem_in>(ptr, array, allocated_size);           
    }
    else
    {
        nrow = nrow_in;
        ncol = ncol_in;
        allocated_size = 0;
        ptr = array;
    }
}

template <typename T, MemType dev>
matrix_base<T, dev>::matrix_base(const int nrow_in, const int ncol_in)
{
    nrow = nrow_in;
    ncol = ncol_in;
    allocated_size = nrow*ncol*sizeof(T);
    buf.get_memory(allocated_size);
    ptr = static_cast<T*>(buf.ptr());
}

template <typename T, MemType dev>
matrix_base<T, dev>::~matrix_base()
{
    // memproc::dealloc<dev>((void *&)ptr, allocated_size);
    buf.free_memory();
}

template <typename T, MemType dev>
T* matrix_base<T, dev>::get_ptr()
{
    return ptr;
}

template <typename T, MemType dev>
const T* matrix_base<T, dev>::get_ptr() const
{
    return ptr;
}

template <typename T, MemType dev>
size_t matrix_base<T, dev>::get_size() const
{
    return allocated_size;
}

template <typename T, MemType dev>
void matrix_base<T, dev>::allocate(const size_t size)
{
    buf.get_memory(size);
    allocated_size = size;
    ptr = static_cast<T*>(buf.ptr());
    // memproc::realloc<dev>((void *&)ptr, allocated_size, size);
}

template <typename T, MemType dev>
matrix_base<T, dev>& matrix_base<T, dev>::operator=(const matrix_base<T, dev>& other)
{
    if (this == &other)
        return *this;
 
    buf.get_memory(other.allocated_size);
    ptr = static_cast<T*>(buf.ptr());
    allocated_size = other.allocated_size;
    nrow = other.nrow;
    ncol = other.ncol;

    memproc::memcopy<dev, dev>(ptr, other.ptr, allocated_size);

    return *this;
}

template <typename T, MemType dev>
std::tuple<int, int> matrix_base<T, dev>::get_dims() const
{
    return std::make_tuple(nrow, ncol);
}
#include <iostream>
template <typename T, MemType dev>
matrix<T, dev> matrix<T, dev>::operator *(const matrix<T, dev>& other) const
{
    std::tuple<int, int> dims = other.get_dims();
    const int A_nrow = nrow;
    const int A_ncol = ncol;
    const int B_ncol = std::get<1>(dims);

    matrix<T, dev> Res(A_nrow, B_ncol);

    const T* A = ptr;
    const T* B = other.get_ptr();
    T* C = Res.get_ptr();

    for (int i = 0; i < A_nrow; i++)
    {
        for (int j = 0; j < B_ncol; j++)
        {
            T sum_value = 0;
            for (int k = 0; k < A_ncol; k++)
            {
                sum_value += A[i * A_ncol + k] * B[k * B_ncol + j];
            }
            C[i * B_ncol + j] = sum_value;
        }
    }
    
    return Res;
}

template class matrix_base<float, MemType::CPU>;
template class matrix_base<double, MemType::CPU>;
template class matrix_base<int, MemType::CPU>;

template void matrix_base<int, MemType::CPU>::set_array<MemType::CPU>(int* array, const int nrow_in, const int ncol_in);
template void matrix_base<float, MemType::CPU>::set_array<MemType::CPU>(float* array, const int nrow_in, const int ncol_in);
template void matrix_base<double, MemType::CPU>::set_array<MemType::CPU>(double* array, const int nrow_in, const int ncol_in);

#ifdef ENABLE_GPU_COMPUTATIONS
template void matrix_base<int, MemType::CPU>::set_array<MemType::GPU>(int* array, const int nrow_in, const int ncol_in);
template void matrix_base<float, MemType::CPU>::set_array<MemType::GPU>(float* array, const int nrow_in, const int ncol_in);
template void matrix_base<double, MemType::CPU>::set_array<MemType::GPU>(double* array, const int nrow_in, const int ncol_in);

template class matrix_base<float, MemType::GPU>;
template class matrix_base<double, MemType::GPU>;
template class matrix_base<int, MemType::GPU>;

template void matrix_base<int, MemType::GPU>::set_array<MemType::GPU>(int* array, const int nrow_in, const int ncol_in);
template void matrix_base<float, MemType::GPU>::set_array<MemType::GPU>(float* array, const int nrow_in, const int ncol_in);
template void matrix_base<double, MemType::GPU>::set_array<MemType::GPU>(double* array, const int nrow_in, const int ncol_in);

template void matrix_base<int, MemType::GPU>::set_array<MemType::CPU>(int* array, const int nrow_in, const int ncol_in);
template void matrix_base<float, MemType::GPU>::set_array<MemType::CPU>(float* array, const int nrow_in, const int ncol_in);
template void matrix_base<double, MemType::GPU>::set_array<MemType::CPU>(double* array, const int nrow_in, const int ncol_in);
#endif

template class matrix<float, MemType::CPU>;
template class matrix<double, MemType::CPU>;
template class matrix<int, MemType::CPU>;
