# Build
Used packages:
  - pybind11 library (for the C++ implementtaion of python module )
  - numpy for tests

CMake is used to build and compile the module.
To specify python, set the `Python_ROOT_DIR` option equal to the path where python is located:
```
cmake -DPython_ROOT_DIR=</path/to/python> ..
```
To enable GPU computations with CUDA C options `INCLUDE_CUDA` and `CMAKE_CUDA_ARCHITECTURES` should be set:
```
cmake -DINCLUDE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<N>..
```
where `CMAKE_CUDA_ARCHITECTURES` corresponds to GPU architecture.

The project also includes unit tests (./tests/test_invalid_args.py) and perfomance tests compared to the numpy matrix multiplication implementation (./tests/perfomance.py). Tests are specified while building, and located at `${CMAKE_CURRENT_BINARY_DIR}/tests/` 
