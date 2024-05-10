import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

def f(x):
    return 4 / (1 + x ** 2)

def rectangles(m):
    n = 2 ** m
    d = 1 / n
    integral = 0

    for i in range(n):
        a = i * d
        b = a + d
        integral += d * f((a + b) / 2)
    
    return integral

def trapezes(m):
    x = 



    n = 2 ** m
    d = 1 / n
    integral = 0

    for i in range(n):
        a = i * d
        b = a + d
        integral += d * (f(a) + f(b)) / 2

    return integral

def simpson(m):
    n = 2 ** m
    d = 1 / n
    integral = 0

    for i in range(n):
        a = i * d
        b = a + d
        c = (a + b) / 2

        x = np.array([a, c, b])
        y = np.array([f(a), f(c), f(b)])
        A = np.vstack([x ** 2, x, np.ones_like(x)]).T
        c1, c2, c3 = np.linalg.solve(A, y)
        integral += c1 * b ** 3 / 3 + c2 * b ** 2 / 2 + c3 * b
        integral -= c1 * a ** 3 / 3 + c2 * a ** 2 / 2 + c3 * a
    
    return integral

def gauss_legendre(n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    t = nodes / 2 + 0.5# t = (1/2)x + 1/2

    integral = np.sum(weights * f(t)) / 2
    print(integral)
    return integral

def zad1():
    m_max = 25
    n = 2 ** m_max + 1
    x = np.linspace(1, n, m_max)

    #rectangles
    err = np.array([abs(rectangles(m) - np.pi) / np.pi for m in range(1, m_max + 1)])
    plt.plot(x, err, label="rectangles method")

    #trapezes
    #err = np.array([abs(trapezes(m) - np.pi) / np.pi for m in range(1, m_max + 1)])
    #plt.plot(x, err, label="trapezes method")

    #simpson
    #err = np.array([abs(simpson(m) - np.pi) / np.pi for m in range(1, m_max + 1)])
    #plt.plot(x, err, label="Simpson method")

    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("relative error")
    plt.legend()
    plt.title("Error comparison for numerical integration")
    plt.show()


zad1()
