import numpy as np
from gradient_descent import *
from line_search import *

def test_function_1(x):
    f = np.sum(np.square(x))
    return f

def test_gradient_function_1(x):
    gf = 2 * x
    return gf

def test_function_rosenbrock(x):
    a = 1
    b = 100
    f = np.sum(np.square(x[::2] - a) + b * np.square(np.square(x[::2]) - x[1::2]))
    return f

def test_gradient_function_rosenbrock(x):
    a = 1 
    b = 100
    gf = np.zeros(x.shape)
    gf[::2] = 2 * (x[::2] - a) + 4 * b * (np.square(x[::2]) - x[1::2]) * x[::2]
    gf[1::2] = -2 * b * (np.square(x[::2]) - x[1::2])
    return gf

if __name__ == "__main__":
    print("Testing function 1:")
    x0 = np.array([1, 1])
    alpha = 1
    print(line_search(test_function_1, test_gradient_function_1, alpha, x0, eps=1e-6))

    print("Testing function Rosenbrock:")
    x0 = np.array([3, 4, 0, 0])
    alpha = 1
    print(line_search(test_function_rosenbrock, test_gradient_function_rosenbrock, alpha, x0, eps=1e-6, max_iter=10000))