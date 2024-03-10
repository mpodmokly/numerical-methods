import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

val = 1

def derivative1(f, x, h):
    return (f(x + h) - f(x)) / h

def derivative2(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def rel_tan(x):
    return (np.tan(x)) ** 2 + 1

def h_min1(x):
    M = np.tan(val) ** 3 + np.tan(val)
    return 2 * np.sqrt(np.finfo(float).eps / abs(2 * (M)))

def h_min2(x):
    M = 6 * np.tan(val) ** 4 + 8 * np.tan(val) ** 2 + 2
    return np.cbrt((3 * np.finfo(float).eps) / abs(M))

def zad1():
    x = np.logspace(0, -16, base=10)
    y = abs(rel_tan(1) - derivative1(np.tan, val, x)) / rel_tan(1)

    h_min = x[np.argmin(y)]
    print("numeryczne h_min = " + str(h_min))
    h_min = h_min1(val)
    print("prawdziwe h_min = " + str(h_min))

    plt.plot(x, y)
    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("h")
    plt.ylabel("Oś Y")
    plt.title("Wykres błędu")
    plt.show()

def zad2():
    np.float32
    n = 225
    sequence1 = np.array([1/3, 1/12])
    for i in range(n - 2):
        sequence1 = np.append(sequence1, 2.25 * sequence1[-1] - 0.5 * sequence1[-2])

    np.float64
    n = 60
    sequence2 = np.array([1/3, 1/12])
    for i in range(n - 2):
        sequence2 = np.append(sequence2, 2.25 * sequence2[-1] - 0.5 * sequence2[-2])

    n = 225
    sequence3 = np.array([Fraction(1, 3), Fraction(1, 12)])
    for i in range(n - 2):
        sequence3 = np.append(sequence3, Fraction(9,4) * sequence3[-1] - Fraction(1,2) * sequence3[-2])

    x = np.linspace(0, n - 1, n)
    real = (4 ** -x) / 3
    err = abs(sequence1 - real) / real
    plt.scatter(x, err)
    err = abs(sequence3 - real) / real
    plt.scatter(x, err)

    n = 60
    x = np.linspace(0, n - 1, n)
    real = (4 ** -x) / 3
    err = abs(sequence2 - real) / real
    plt.scatter(x, err)

    plt.yscale("log")
    plt.xlabel("k")
    plt.show()

zad1()
