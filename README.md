# Build
Used packages:
  - pybind11 library (for the C++ implementtaion of python module )
  - numpy, timeit, pytest for tests

CMake (minimum version is 3.19) is used to build and compile the module.
To specify python, set the `Python_ROOT_DIR` option equal to the path where python is located:
```
cmake -DPython_ROOT_DIR=</path/to/python> ..
```
To enable GPU computations options `INCLUDE_CUDA` and `CMAKE_CUDA_ARCHITECTURES` should be set:
```
cmake -DINCLUDE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<N>..
```
where `CMAKE_CUDA_ARCHITECTURES` corresponds to the GPU architecture.

The project also includes unit tests (./tests/test_invalid_args.py) and tests of the perfomance comparison between current library and the numpy matrix multiplication implementation (./tests/perfomance.py). Tests are specified while building and located at `${CMAKE_CURRENT_BINARY_DIR}/tests/`.
