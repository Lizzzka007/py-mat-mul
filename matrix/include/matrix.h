#pragma once
#include <tuple>
#include "memory-processing.h"

template <typename T, MemType dev>
class matrix_base
{
public:
    // int stride_row, stride_col;
    int nrow, ncol;
    size_t allocated_size;
    T *ptr;
    
    matrix_base();
    matrix_base(const int nrow, const int ncol);

    template <MemType mem_in>
    void set_array(T* array, const int nrow, const int ncol);
    
    ~matrix_base();

    std::tuple<int, int> get_dims() const;
    void allocate(const size_t size);
    T* get_ptr();
    size_t get_size() const;
    const T* get_ptr() const;
    matrix_base<T, dev>& operator=(const matrix_base<T, dev>& other);

    // void print_array() const;
};

template <typename T, MemType dev>
class matrix: public matrix_base<T, dev>
{
private:
    // using matrix_base<T, dev>::stride_row;
    // using matrix_base<T, dev>::stride_col;
    using matrix_base<T, dev>::nrow;
    using matrix_base<T, dev>::ncol;
    using matrix_base<T, dev>::allocated_size;
    using matrix_base<T, dev>::ptr;
public:
    matrix() : matrix_base<T, dev>(){}
    matrix(const int nrow, const int ncol) : matrix_base<T, dev>(nrow, ncol){}
    ~matrix() {}

    matrix<T, dev> operator *(const matrix<T, dev>& other) const; 
};

#ifdef ENABLE_GPU_COMPUTATIONS
template <typename T>
class matrix<T, MemType::GPU>: public matrix_base<T, MemType::GPU>
{
private:
    // using matrix_base<T, MemType::GPU>::stride_row;
    // using matrix_base<T, MemType::GPU>::stride_col;
    using matrix_base<T, MemType::GPU>::nrow;
    using matrix_base<T, MemType::GPU>::ncol;
    using matrix_base<T, MemType::GPU>::allocated_size;
    using matrix_base<T, MemType::GPU>::ptr;
public:
    matrix() : matrix_base<T, MemType::GPU>(){}
    matrix(const int nrow, const int ncol) : matrix_base<T, MemType::GPU>(nrow, ncol){}
    ~matrix() {}

    matrix<T, MemType::GPU> operator *(const matrix<T, MemType::GPU>& other) const; 
};
#endif