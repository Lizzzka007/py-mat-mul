#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <typename T>
void mat_mul_typed( py::array_t<T> A,  py::array_t<T> B, const int useGpu); 