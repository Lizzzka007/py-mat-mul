# Build
CMake is used to build and compile the module.
To specify python, set the `Python_ROOT_DIR` option equal to the path where python is located:
@bash
```
cmake -DPython_ROOT_DIR=</path/to/python> ..
```
To enable GPU computations with CUDA C options `INCLUDE_CUDA` and `CMAKE_CUDA_ARCHITECTURES` shoul be set:
@bash
```
cmake -DINCLUDE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<N>..
```
where `CMAKE_CUDA_ARCHITECTURES` corresponds to GPU architecture.

The project also includes unit tests (./tests/test_invalid_args.py) and perfomance tests compared to the numpy matrix multiplication implementation (./tests/perfomance.py).
