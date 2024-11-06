#include <iostream>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "matrix_lib.h"
#include "MemoryProcessing.h"

namespace py = pybind11;

void process_object(const py::object object, matrix_lib::pyobj_info& info)
{
    if (py::isinstance<py::array>(object)) 
    {
        py::array array = object.cast<py::array>();
        py::buffer_info py_info = array.request();
        if(py_info.ndim != 2)
        {
            throw std::invalid_argument("Input must be a 2D array.");
        }

        info.nrow = py_info.shape[0];
        info.ncol = py_info.shape[1];
        info.format = py_info.format;
        info.mem_type = MemType::CPU;
        info.ptr = py_info.ptr;
    }
    else 
    {
        throw std::invalid_argument("Objects representing matrices must be numpy arrays.");
    }
}

void is_valid_matrices(matrix_lib::pyobj_info& info_A, matrix_lib::pyobj_info& info_B)
{
    if (info_A.format != info_B.format)
    {
        throw std::invalid_argument("Incompatible data type!");
    }
    if(info_A.ncol != info_B.nrow)
    {
        throw std::invalid_argument("Matrix dimensions do not match.");
    }
    if (info_A.nrow == 0)
    {
        std::string error_message = "Matrix is empty. Row number of first matrix is " + 
            std::to_string(info_A.nrow) + ".";
        throw std::invalid_argument(error_message.c_str()); 
    }
    if (info_A.ncol == 0)
    {
        std::string error_message = "Matrix is empty. Column number of first matrix is " + 
            std::to_string(info_A.ncol) + ".";
        throw std::invalid_argument(error_message.c_str()); 
    }
    if (info_B.nrow == 0)
    {
        std::string error_message = "Matrix is empty. Column number of second matrix is " + 
            std::to_string(info_B.nrow) + ".";
        throw std::invalid_argument(error_message.c_str()); 
    }
    if (info_B.ncol == 0)
    {
        std::string error_message = "Matrix is empty. Column number of second matrix is " + 
            std::to_string(info_B.ncol) + ".";
        throw std::invalid_argument(error_message.c_str()); 
    }
}

py::object mat_mul( py::object A_obj, py::object B_obj, const int useGpu)
{
#ifndef ENABLE_GPU_COMPUTATIONS
    if (useGpu)
    {
        throw std::invalid_argument("GPU computations are not enabled");
    }
#endif

    matrix_lib::pyobj_info A_info, B_info;
    try {
        process_object(A_obj, A_info);
        process_object(B_obj, B_info);
        is_valid_matrices(A_info, B_info);
    }
    catch (const std::invalid_argument& e) 
    {
        std::string error_message = e.what();
        py::set_error(PyExc_ValueError, error_message.c_str());
        throw py::error_already_set();
    }

    py::object Res_obj;

    if (A_info.format == py::format_descriptor<int>::format())
    {
        py::array_t<int> py_array_result  = py::array_t<int>({A_info.nrow, B_info.ncol}, 
                                                            {B_info.ncol * sizeof(int), sizeof(int)});
        py::buffer_info result_info = py_array_result.request();
        int * ptr = static_cast<int *>(result_info.ptr);

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
        
        Res_obj = py::cast<py::object>(py_array_result);
    }
    else if (A_info.format == py::format_descriptor<float>::format())
    {
        py::array_t<float> py_array_result  = py::array_t<float>({A_info.nrow, B_info.ncol}, 
                                                            {B_info.ncol * sizeof(float), sizeof(float)});
        py::buffer_info result_info = py_array_result.request();
        float * ptr = static_cast<float *>(result_info.ptr);
    
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

        Res_obj = py::cast<py::object>(py_array_result);
    }
    else if (A_info.format == py::format_descriptor<double>::format())
    {
        py::array_t<double> py_array_result  = py::array_t<double>({A_info.nrow, B_info.ncol}, 
                                                            {B_info.ncol * sizeof(double), sizeof(double)});
        py::buffer_info result_info = py_array_result.request();
        double * ptr = static_cast<double *>(result_info.ptr);

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

        Res_obj = py::cast<py::object>(py_array_result);
    }
    else
    {
        py::set_error(PyExc_ValueError, "Unsupported data type. Available data types: int, float32, float64");
        throw py::error_already_set();
    }

    return Res_obj;
}

PYBIND11_MODULE(py_mat_mul, pmm) {
    pmm.doc() = "C++ module for matrix multiplication"; 
    pmm.def("mat_mul", &mat_mul, "A function that implements matrix multiplication of 2D numpy objects ");
}