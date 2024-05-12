import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.integrate import simps

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
    n = 2 ** m
    x = np.linspace(0, 1, n)
    y = f(x)

    integral = trapz(y, x)
    return integral

def simpson(m):
    n = 2 ** m
    x = np.linspace(0, 1, n)
    y = f(x)

    integral = simps(y, x)
    return integral

def gauss_legendre(n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    t = nodes / 2 + 0.5

    integral = np.sum(weights * f(t)) / 2
    return integral

def zad1():
    m_max = 25
    n = 2 ** m_max
    x = np.linspace(1, n, m_max)

    err = np.array([abs(rectangles(m) - np.pi) / np.pi\
                    for m in range(1, m_max + 1)])
    plt.plot(x, err, label="rectangles method")

    err = np.array([abs(trapezes(m) - np.pi) / np.pi\
                    for m in range(1, m_max + 1)])
    plt.plot(x, err, label="trapezes method")

    err = np.array([abs(simpson(m) - np.pi) / np.pi\
                    for m in range(1, m_max + 1)])
    plt.plot(x, err, label="Simpson method")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("relative error")
    plt.legend()
    plt.title("Error comparison for numerical integration")
    plt.show()

def zad2():
    n_max = 15
    x = np.linspace(1, n_max, n_max)

    err = np.array([abs(gauss_legendre(n) - np.pi) / np.pi\
                    for n in range(1, n_max + 1)])
    
    plt.plot(x, err)
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("relative error")
    plt.title("Relative error for Gauss-Legendre method")
    plt.show()


zad1()
