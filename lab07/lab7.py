import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad_vec
from scipy.integrate import trapz
from scipy.integrate import simps

EVALS = 14

def f1(x):
    return 4 / (1 + x ** 2)

def f2(x):
    return np.sqrt(x) * np.log(x)

def f3(x):
    a = 0.001
    b = 0.004
    return 1 / ((x - 0.3) ** 2 + a) + 1 / ((x - 0.9) ** 2 + b) - 6

def rectangles(m, f):
    n = 2 ** m + 1
    d = 1 / n
    integral = 0

    for i in range(n):
        a = i * d
        b = a + d
        integral += d * f((a + b) / 2)
    
    return integral

def trapezes(m, f):
    n = 2 ** m + 1
    x = np.linspace(0, 1, n)

    if f == f2:
        x[0] = 1
        y = f(x)
        y[0] = 0
        x[0] = 0
    else:
        y = f(x)

    integral = trapz(y, x)
    return integral

def simpson(m, f):
    n = 2 ** m + 1
    x = np.linspace(0, 1, n)

    if f == f2:
        x[0] = 1
        y = f(x)
        y[0] = 0
        x[0] = 0
    else:
        y = f(x)

    integral = simps(y, x)
    return integral

def gauss_legendre(n, f):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    t = nodes / 2 + 0.5

    integral = np.sum(weights * f(t)) / 2
    return integral

def quad_vec_err(f, q, ans):
    eps = np.logspace(0, -EVALS, EVALS + 1)
    x = np.array([])
    y = np.array([])

    a = 0
    if f == f2 and q == "trapezoid":
        a = 1e-14

    for i in range(EVALS + 1):
        result, _, info = quad_vec(f, a, 1, epsrel=eps[i], quadrature=q,\
                                full_output=True)
        x = np.append(x, info.neval)
        y = np.append(y, abs(result - ans) / abs(ans))
    
    return x, y

def zad1():
    ans = np.pi
    x_trapz, y_trapz = quad_vec_err(f1, "trapezoid", ans)
    x_gk, y_gk = quad_vec_err(f1, "gk15", ans)
    plt.plot(x_trapz, y_trapz, label="adaptive trapezes")
    plt.plot(x_gk, y_gk, label="Gauss-Kronrod method")

    m_max = 16
    n = 2 ** m_max

    x = np.linspace(1, n, m_max)
    err = np.array([abs(rectangles(m, f1) - ans) / abs(ans)\
                    for m in range(1, m_max + 1)])
    plt.plot(x, err, label="rectangles method")
    err = np.array([abs(trapezes(m, f1) - ans) / abs(ans)\
                    for m in range(1, m_max + 1)])
    plt.plot(x, err, label="trapezes method")
    err = np.array([abs(simpson(m, f1) - ans) / abs(ans)\
                    for m in range(1, m_max + 1)])
    plt.plot(x, err, label="Simpson method")
    err = np.array([abs(gauss_legendre(m, f1) - ans) / abs(ans)\
                    for m in range(1, m_max + 1)])
    plt.plot(x, err, label="Gauss-Legendre method")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n")
    plt.ylabel("relative error")
    plt.title("Error comparison for numerical integration")
    plt.legend()
    plt.show()

def zad2_int(x0, a):
    result = np.arctan((1 - x0) / np.sqrt(a)) + np.arctan(x0 / np.sqrt(a))
    return result / np.sqrt(a)

def zad2(f):
    m_max = 16
    n = 2 ** m_max + 1

    if f == f1:
        ans = np.pi
    elif f == f2:
        ans = - 4 / 9
    elif f == f3:
        a = 0.001
        b = 0.004

        ans = zad2_int(0.3, a)
        ans += zad2_int(0.9, b)
        ans -= 6
    else:
        print("Function error")
        return

    x = np.linspace(1, n, m_max)
    err = np.array([abs(rectangles(m, f) - ans) / abs(ans)\
                    for m in range(1, m_max + 1)])
    plt.plot(x, err, label="rectangles method")
    err = np.array([abs(trapezes(m, f) - ans) / abs(ans)\
                    for m in range(1, m_max + 1)])
    plt.plot(x, err, label="trapezes method")
    err = np.array([abs(simpson(m, f) - ans) / abs(ans)\
                    for m in range(1, m_max + 1)])
    plt.plot(x, err, label="Simpson method")
    err = np.array([abs(gauss_legendre(m, f) - ans) / abs(ans)\
                    for m in range(1, m_max + 1)])
    plt.plot(x, err, label="Gauss-Legendre method")

    x_trapz, y_trapz = quad_vec_err(f, "trapezoid", ans)
    x_gk, y_gk = quad_vec_err(f, "gk15", ans)
    plt.plot(x_trapz, y_trapz, label="adaptive trapezes")
    plt.plot(x_gk, y_gk, label="Gauss-Kronrod method")

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("n")
    plt.ylabel("relative error")
    plt.title("Error comparison for numerical integration")
    plt.legend()
    plt.show()


#zad1()
zad2(f3)
