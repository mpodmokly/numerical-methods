import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

val = 1
eps = np.finfo(float).eps

def forward_difference_method(f, x, h):
    return (f(x + h) - f(x)) / h

def central_difference_method(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def rel_tan(x):
    return (np.tan(x)) ** 2 + 1

def M_forward(x):
    return abs(np.tan(x) ** 3 + np.tan(x))

def M_central(x):
    return abs(6 * np.tan(x) ** 4 + 8 * np.tan(x) ** 2 + 2)

def h_min_forward(x):
    return 2 * np.sqrt(eps / M_forward(x))

def h_min_central(x):
    return np.cbrt((3 * eps) / M_central(x))

def zad1():
    h = np.logspace(0, -16, base=10)
    computational_err_forward = abs(rel_tan(1) - forward_difference_method(np.tan, val, h)) / rel_tan(1)
    computational_err_central = abs(rel_tan(1) - central_difference_method(np.tan, val, h)) / rel_tan(1)
    truncation_err_forward = M_forward(val) * h / 2
    truncation_err_central = M_forward(val) * (h ** 2) / 6
    rounding_err_forward = 2 * eps / h
    rounding_err_central = eps / h

    h_min = h[np.argmin(computational_err_central)]
    print("numerical h_min = " + str(h_min))
    h_min = h_min_central(val)
    print("true h_min = " + str(h_min))

    plt.plot(h, computational_err_central, label="computational error")
    plt.plot(h, truncation_err_central, label="truncation error")
    plt.plot(h, rounding_err_central, label="rounding error")
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("h")
    plt.ylabel("Error values")
    plt.title("Errors in numerical derivatives")
    plt.legend()
    plt.show()

def zad2():
    n = 225
    sequence1 = np.float32(np.array([1/3, 1/12]))
    for i in range(n - 2):
        sequence1 = np.float32(np.append(sequence1, np.float32(2.25) * sequence1[-1] - np.float32(0.5) * sequence1[-2]))

    x = np.linspace(0, n - 1, n)
    plt.plot(x, sequence1, label="x32, n = 225")

    real = (4 ** -x) / 3
    err = abs(sequence1 - real) / real
    #plt.plot(x, err, label="x32, n = 225")

    
    n = 60
    sequence2 = np.array([1/3, 1/12])
    for i in range(n - 2):
        sequence2 = np.append(sequence2, 2.25 * sequence2[-1] - 0.5 * sequence2[-2])

    x = np.linspace(0, n - 1, n)
    plt.plot(x, sequence2, label="x64, n = 60")

    real = (4 ** -x) / 3
    err = abs(sequence2 - real) / real
    #plt.plot(x, err, label="x64, n = 60")

    n = 225
    sequence3 = np.array([Fraction(1,3), Fraction(1,12)])
    for i in range(n - 2):
        sequence3 = np.append(sequence3, Fraction(9,4) * sequence3[-1] - Fraction(1,2) * sequence3[-2])

    x = np.linspace(0, n - 1, n)
    plt.plot(x, sequence3, label="fractions, n=225")

    real = (4 ** -x) / 3
    err = abs(sequence3 - real) / real
    #plt.plot(x, err, label="fractions, n=225")

    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("Error values")
    plt.legend()
    plt.show()

zad2()
