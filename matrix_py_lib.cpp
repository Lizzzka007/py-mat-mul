#include <iostream>
#include "matrix_py_lib.h"
#include "matrix_lib.h"
#include "memory-processing.h"
// #ifdef ENABLE_GPU_COMPUTATIONS
// #include <cupy/array.h>
// #endif
// #include <frameobject.h>

void process_object(const py::object object, matrix_lib::pyobj_info& info)
{
    if (py::isinstance<py::array>(object)) 
    {
        py::array array = object.cast<py::array>();
        py::buffer_info py_info = array.request();
        if(py_info.ndim != 2)
        {
            throw std::runtime_error("Input must be a 2D array.");
        }

        info.nrow = py_info.shape[0];
        info.ncol = py_info.shape[1];
        info.format = py_info.format;
        info.mem_type = MemType::CPU;
        info.ptr = py_info.ptr;
    }
    else 
    {
        throw std::runtime_error("Input must be a NumPy or CuPy array.");
    }
}

void is_valid_matrices(matrix_lib::pyobj_info& info_A, matrix_lib::pyobj_info& info_B)
{
    if (info_A.format != info_B.format)
    {
        throw std::runtime_error("Incompatible data type!");
    }
    if(info_A.ncol != info_B.nrow)
    {
        throw std::invalid_argument("matrix dimensions do not match");
    }

}

py::object mat_mul( py::object A_obj, py::object B_obj, const int useGpu)
{
    // TODO: обработать throw
    matrix_lib::pyobj_info A_info, B_info;
    process_object(A_obj, A_info);
    process_object(B_obj, B_info);

    // TODO: обработать throw
    is_valid_matrices(A_info, B_info);
    py::object Res_obj;

    if (A_info.format == py::format_descriptor<int>::format())
    {
        std::cout << "int" << std::endl;
        int * ptr;
        py::array_t<int> py_array_result  = py::array_t<int>({A_info.nrow, B_info.ncol}, 
                                                            {B_info.ncol * sizeof(int), sizeof(int)},
                                                            ptr);

#ifdef ENABLE_GPU_COMPUTATIONS
        if (useGpu)
        {
            matrix_lib::mat_mul<int, MemType::GPU>(A_info, B_info, ptr);
        }
    else
#endif
        {
            matrix_lib::mat_mul<int, MemType::CPU>(A_info, B_info, ptr);
        }
        
        Res_obj = (py::object)py_array_result;
    }
    else if (A_info.format == py::format_descriptor<float>::format())
    {
        std::cout << "float" << std::endl;
        float * ptr;
    
#ifdef ENABLE_GPU_COMPUTATIONS
        if (useGpu)
        {
            matrix_lib::mat_mul<float, MemType::GPU>(A_info, B_info, ptr);
        }
    else
#endif
        {
            matrix_lib::mat_mul<float, MemType::CPU>(A_info, B_info, ptr);
        }

        py::array_t<float> py_array_result  = py::array_t<float>({A_info.nrow, B_info.ncol}, 
                                                            {B_info.ncol * sizeof(float), sizeof(float)},
                                                            ptr);
        Res_obj = (py::object)py_array_result;

    }
    else
    {
        std::cout << "double" << std::endl;
        double * ptr;
        py::array_t<double> py_array_result  = py::array_t<double>({A_info.nrow, B_info.ncol}, 
                                                            {B_info.ncol * sizeof(double), sizeof(double)},
                                                            ptr);
#ifdef ENABLE_GPU_COMPUTATIONS
        if (useGpu)
        {
            matrix_lib::mat_mul<double, MemType::GPU>(A_info, B_info, ptr);
        }
    else
#endif
        {
            matrix_lib::mat_mul<double, MemType::CPU>(A_info, B_info, ptr);
        }

        Res_obj = (py::object)py_array_result;
    }

    return Res_obj;
}

PYBIND11_MODULE(py_mat_mul, pmm) {
    pmm.doc() = "My module documentation"; // Документация модуля
    pmm.def("mat_mul", &mat_mul, "A function that calls a function from B");
}