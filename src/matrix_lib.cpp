#include "matrix_lib.h"
#include "MemoryProcessing.h"

template <typename T, MemType dev_run>
void matrix_lib::mat_mul( pyobj_info& info_A, pyobj_info& info_B, T*& Res_ptr )
{
    static matrix<T, dev_run> A_mat;
    static matrix<T, dev_run> B_mat;

    T* A_ptr = static_cast<T*>(info_A.ptr);
    T* B_ptr = static_cast<T*>(info_B.ptr);

    if(info_A.mem_type == MemType::CPU)
        A_mat.template set_array<MemType::CPU>(A_ptr, info_A.nrow, info_A.ncol);
#ifdef ENABLE_GPU_COMPUTATIONS
    else
        A_mat.template set_array<MemType::GPU>(A_ptr, info_A.nrow, info_A.ncol);
#endif

    if(info_B.mem_type == MemType::CPU)
        B_mat.template set_array<MemType::CPU>(B_ptr, info_B.nrow, info_B.ncol);
#ifdef ENABLE_GPU_COMPUTATIONS
    else
        B_mat.template set_array<MemType::GPU>(B_ptr, info_B.nrow, info_B.ncol);
#endif

    matrix<T, dev_run> Res_mat = A_mat * B_mat;
    T* Res_mat_ptr = Res_mat.get_ptr();
    memproc::memcopy<MemType::CPU, dev_run>(Res_ptr, Res_mat_ptr, Res_mat.get_size());
}

template void matrix_lib::mat_mul<int, MemType::CPU>( pyobj_info& info_A, pyobj_info& info_B, int*& Res_ptr);
template void matrix_lib::mat_mul<float, MemType::CPU>( pyobj_info& info_A, pyobj_info& info_B, float*& Res_ptr);
template void matrix_lib::mat_mul<double, MemType::CPU>( pyobj_info& info_A, pyobj_info& info_B, double*& Res_ptr);

#ifdef ENABLE_GPU_COMPUTATIONS
template void matrix_lib::mat_mul<int, MemType::GPU>( pyobj_info& info_A, pyobj_info& info_B, int*& Res_ptr );
template void matrix_lib::mat_mul<float, MemType::GPU>( pyobj_info& info_A, pyobj_info& info_B, float*& Res_ptr );
template void matrix_lib::mat_mul<double, MemType::GPU>( pyobj_info& info_A, pyobj_info& info_B, double*& Res_ptr );
#endif