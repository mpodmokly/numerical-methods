import numpy as np
from scipy.optimize import newton

def f1(x):
    return x ** 3 - 5 * x

def f2(x):
    return x ** 3 - 3 * x + 1

def f3(x):
    return 2 - x ** 5

def f4(x):
    return x ** 4 - 4.29 * x ** 2 - 5.29

def find_roots(f, est):
    result = np.array([])
    for x0 in est:
        result = np.append(result, newton(f, x0))
    
    return result

def zad1():
    est = [-2, 0, 2]
    result = find_roots(f1, est)
    print(result)

    est = [-2, 0, 2]
    result = find_roots(f2, est)
    print(result)

    est = [1]
    result = find_roots(f3, est)
    print(result)

    est = [-2, 2]
    result = find_roots(f4, est)
    print(result)

