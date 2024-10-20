import pytest
import numpy as np
import py_mat_mul
from test_config import GPU_ENABLED

def test_empty_matrix():
    a = np.ndarray(shape=(9, 0), dtype=np.float32)
    b = np.ndarray(shape=(3, 6), dtype=np.float32)

    try:
        py_mat_mul.mat_mul(a, b, 0)
        assert 1 == 0, "Should raise ValueError"
    except ValueError as e:
        print(f"Test on empty matrix is passed. Error message is: {e}")

def test_non_matrix_input():
    a = 6
    b = np.ndarray(shape=(3, 6), dtype=np.float32)

    try:
        py_mat_mul.mat_mul(a, b, 0)
        assert 1 == 0, "Should raise ValueError"
    except ValueError as e:
        print(f"Test on non matrix input is passed. Error message is: {e}")

def test_different_matrix_datatypes():
    a = np.ndarray(shape=(8, 3), dtype=np.float64)
    b = np.ndarray(shape=(3, 6), dtype=np.float32)

    try:
        py_mat_mul.mat_mul(a, b, 0)
        assert 1 == 0, "Should raise ValueError"
    except ValueError as e:
        print(f"Test on different datatypes is passed. Error message is: {e}")

def test_matrix_dimension_compatible():
    a = np.ndarray(shape=(8, 4), dtype=np.float32)
    b = np.ndarray(shape=(3, 6), dtype=np.float32)

    try:
        py_mat_mul.mat_mul(a, b, 0)
        assert 1 == 0, "Should raise ValueError"
    except ValueError as e:
        print(f"Test on incompatible matrix dimensions is passed. Error message is: {e}")

def test_matrix_one_dimensional():
    a = np.zeros(8, dtype=np.float32)
    b = np.zeros(9, dtype=np.float32)

    try:
        py_mat_mul.mat_mul(a, b, 0)
        assert 1 == 0, "Should raise ValueError"
    except ValueError as e:
        print(f"Test on one dimensional matrix is passed. Error message is: {e}")

@pytest.mark.skipif(GPU_ENABLED==True, reason="Skipping tests because if_do is False")
def test_call_gpu_if_not_enabled():
    a = np.ndarray(shape=(8, 3), dtype=np.float32)
    b = np.ndarray(shape=(3, 6), dtype=np.float32)

    try:
        py_mat_mul.mat_mul(a, b, 1)
        assert 1 == 0, "Should raise ValueError"
    except ValueError as e:
        print(f"Test on GPU is passed. Error message is: {e}")