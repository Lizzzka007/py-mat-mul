import numpy as np
import py_mat_mul
from test_config import GPU_ENABLED
import time

def perfomance_test():
    global GPU_ENABLED

    print("Test info: field 'Speedup' is equal to numpy computation time / cxx library computation time ")
    print("Test are provided for square matrices of size 2^i")
    print("CPU performance test start.")
    
    n = 10
    for i in range(1, n):
        square_size = 2**i
        a = np.ndarray(shape=(square_size, square_size), dtype=np.float32)
        b = np.ndarray(shape=(square_size, square_size), dtype=np.float32)

        start = time.time()
        np.matmul(a, b)
        end = time.time()
        numpy_time = end - start

        start = time.time()
        py_mat_mul.mat_mul(a, b, 0)
        end = time.time()
        cxx_time = end - start

        speedup = (cxx_time/numpy_time)

        print("Speedup: {:.2f} times, i = {}".format(speedup, i))

    if GPU_ENABLED:
        print("GPU performance test start.")

        for i in range(n):
            square_size = 2**i
            a = np.ndarray(shape=(square_size, square_size), dtype=np.float32)
            b = np.ndarray(shape=(square_size, square_size), dtype=np.float32)

            start = time.time()
            py_mat_mul.mat_mul(a, b, 1)
            end = time.time()
            cxx_time = end - start

            start = time.time()
            np.matmul(a, b)
            end = time.time()
            numpy_time = end - start

            speedup = cxx_time / numpy_time

            print("Speedup: {:.2f} times, i = {}".format(speedup, i))

if __name__ == '__main__':
    perfomance_test()