import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

def zad1():
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

def zad2(plot = False):
    def g1(x):
        return (x ** 2 + 2) / 3
    def g2(x):
        return np.sqrt(3 * x - 2)
    def g3(x):
        return 3 - 2 / x
    def g4(x):
        return (x ** 2 - 2) / (2 * x - 3)
    
    def g1_prim(x):
        return (2 / 3) * x
    def g2_prim(x):
        return 3 / (2 * np.sqrt(3 * x - 2))
    def g3_prim(x):
        return 2 / x ** 2
    def g4_prim(x):
        return (2 * x ** 2 - 6 * x + 4) / (4 * x ** 2 - 12 * x + 9)

    def root_newton(f, f_prim, x0, k, true_x, err = False):
        eps = [abs(x0 - true_x)]
        x = x0
        y = [abs(x - true_x) / true_x]

        for i in range(k):
            x = x - f(x) / f_prim(x)
            eps.append(abs(x - true_x))
            y.append(abs(x - true_x) / true_x)

        if err:
            return y
        
        for i in range(len(eps)):
            if i > 0 and i < len(eps) - 1:
                r = np.log(eps[i] / eps[i + 1]) / np.log(eps[i - 1] / eps[i])
                print(r)
        
        return x
    
    
    true_x = 2
    print(abs(g1_prim(true_x)))
    print(abs(g2_prim(true_x)))
    print(abs(g3_prim(true_x)))
    print(abs(g4_prim(true_x)))

    x0 = 4
    k = 10
    y = root_newton(g1, g1_prim, x0, k, true_x, plot)
    plt.plot(np.linspace(0, k, k + 1), y)
    y = root_newton(g4, g4_prim, x0, k, true_x, plot)
    plt.plot(np.linspace(0, k, k + 1), y)

    k = 6
    y = root_newton(g3, g3_prim, x0, k, true_x, plot)
    plt.plot(np.linspace(0, k, k + 1), y)

    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("relative error")
    plt.title("Relative error for Newton method")
    plt.show()

def zad3():
    def f1(x):
        return x ** 3 - 2 * x - 5
    def f2(x):
        return np.exp(-x) - x
    def f3(x):
        return x * np.sin(x) - 1
    
    def f1_prim(x):
        return 3 * x ** 2 - 2
    def f2_prim(x):
        return -np.exp(-x) - 1
    def f3_prim(x):
        return np.sin(x) + x * np.cos(x)
    
    def root_newton(f, f_prim, x0, true_x, digits):
        eps = [-np.log10(abs((x0 - true_x) / true_x))]
        x = x0
        k = 0

        while eps[-1] < digits:
            x = x - f(x) / f_prim(x)
            eps.append(-np.log10(abs((x - true_x) / true_x)))
            k += 1
        
        if digits == 6:
            print(f"{f.__name__} 24 bits - k = {k}")
        else:
            print(f"{f.__name__} 53 bits - k = {k}")
    
    # 24 bit - 6 digits
    # 53 bit - 14 digits
    digits1 = 6
    digits2 = 14

    x0 = 2
    true_x = 2.09455148154233
    root_newton(f1, f1_prim, x0, true_x, digits1)
    root_newton(f1, f1_prim, x0, true_x, digits2)

    x0 = 0
    true_x = 0.56714329040978387300
    root_newton(f2, f2_prim, x0, true_x, digits1)
    root_newton(f2, f2_prim, x0, true_x, digits2)

    x0 = 1
    true_x = 1.11415714087193008730
    root_newton(f3, f3_prim, x0, true_x, digits1)
    root_newton(f3, f3_prim, x0, true_x, digits2)

def zad4():
    def f(x):
        return x ** 4 + x ** 2 - 1
    def f_prim(x):
        return 4 * x ** 3 + 2 * x
    
    def root_newton(f, f_prim, x0, k):
        x = x0
        for _ in range(k):
            x = x - f(x) / f_prim(x)
        
        return x
    
    x0 = 1
    k = 4
    x = root_newton(f, f_prim, x0, k)

    true_x = np.sqrt(0.5 * np.sqrt(5) - 0.5)
    eps = abs((x - true_x) / true_x)
    print(eps)


zad4()
