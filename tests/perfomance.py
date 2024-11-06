import numpy as np
import py_mat_mul
from test_config import GPU_ENABLED
import timeit

def perfomance_test():
    global GPU_ENABLED
    mult_exec_n = 10
    numpy_times = []

    print("Test info: field 'Speedup' is equal to (average numpy computation time) / (average cxx library computation time) ")
    print("Tests are provided for square matrices of size 2^i")
    print("CPU performance test start.")
    
    n = 11
    # for step in range(1, n):
    for step in range(n, 0, -1):
        square_size = 2**step
        a = np.ndarray(shape=(square_size, square_size), dtype=np.float32)
        b = np.ndarray(shape=(square_size, square_size), dtype=np.float32)

        for i in range(square_size):
            for j in range(square_size):
                a[i][j] = np.float32(i + j) / np.float32(1000.0)
                b[i][j] = np.float32(i - j) / np.float32(1000.0)

        numpy_time = timeit.timeit(lambda: np.matmul(a, b), number=mult_exec_n)
        numpy_time /= mult_exec_n

        if GPU_ENABLED:
            numpy_times.append(numpy_time)

        cxx_time = timeit.timeit(lambda: py_mat_mul.mat_mul(a, b, 0), number=mult_exec_n)
        cxx_time /= mult_exec_n

        speedup = (numpy_time/cxx_time)

        print("Speedup: {:.4f} times, i = {}".format(speedup, step))

    if GPU_ENABLED:
        print("GPU performance test start.")

        # for step, numpy_time in zip(range(1, n), numpy_times):
        for step, numpy_time in zip(range(n, 0, -1), numpy_times):
            square_size = 2**step
            a = np.ndarray(shape=(square_size, square_size), dtype=np.float32)
            b = np.ndarray(shape=(square_size, square_size), dtype=np.float32)

            for i in range(square_size):
                for j in range(square_size):
                    a[i][j] = np.float32(i + j) / np.float32(1000.0)
                    b[i][j] = np.float32(i - j) / np.float32(1000.0)

            cxx_time = timeit.timeit(lambda: py_mat_mul.mat_mul(a, b, 1), number=mult_exec_n)
            cxx_time /= mult_exec_n

            speedup = (numpy_time/cxx_time)

            print("Speedup: {:.4f} times, i = {}".format(speedup, step))

if __name__ == '__main__':
    perfomance_test()