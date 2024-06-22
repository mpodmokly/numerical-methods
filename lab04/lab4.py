import numpy as np
import matplotlib.pyplot as plt

def lagrange(points, vals, x):
    n = len(points)
    result = 0

    for i in range(n):
        product = 1
        for j in range(n):
            if j != i:
                product *= (x - points[j]) / (points[i] - points[j])
        result += product * vals[i]
    
    return result

def splines(n, func, x):
    if func == 1:
        f = f1
        shift = 1
        l = 2
    else:
        f = f2
        shift = 0
        l = 2 * np.pi

    a = (x + shift) // (l / (n - 1)) * (l / (n - 1)) - shift
    b = a + (l / (n - 1))

    points = np.linspace(a, b, 4)
    vals = f(points)

    return lagrange(points, vals, x)

def chebyshev_nodes(n, func):
    chebyshev = np.array([])

    for i in range(n + 1):
        chebyshev = np.append(chebyshev, np.cos((2 * i + 1) / (2 * (n + 1)) * np.pi))
    
    if func != 1:
        for i in range(n + 1):
            chebyshev[i] = 2 * np.pi * (chebyshev[i] + 1) / 2
    
    return chebyshev

def f1(x):
    return 1 / (1 + 25 * x * x)

def f2(x):
    return np.exp(np.cos(x))

def zad1():
    n = 12

    x = np.linspace(-1, 1, 1000)
    y = f1(x)
    plt.plot(x, y, label="Runge's function")

    points = np.linspace(-1, 1, n)
    vals = f1(points)
    plt.scatter(points, vals, color="orange")

    y = lagrange(points, vals, x)
    plt.plot(x, y, color="orange", label="Lagrange")

    y = splines(n, 1, x)
    plt.plot(x, y, color="red", label="splines")

    chebyshev = chebyshev_nodes(n, 1)
    che_vals = f1(chebyshev)

    plt.scatter(chebyshev, che_vals, color="green")
    y = lagrange(chebyshev, che_vals, x)
    plt.plot(x, y, color="green", label="Chebyshev")

    plt.xlabel("Arguments")
    plt.ylabel("Values")
    plt.title("Runge's function interpolation")
    plt.legend()
    plt.show()

def error_plot(func):
    if func == 1:
        a = -1
        b = 1
        f = f1
        title = "Error comparison for Runge's function"
    else:
        a = 0
        b = 2 * np.pi
        f = f2
        title = "Error comparison for exponentional function"

    evaluation = np.random.uniform(a, b, 500)
    x = np.linspace(4, 50, 47)
    y = np.array([])

    for n in range(4, 51):
        points = np.linspace(a, b, n)
        vals = f(points)

        error_norm = np.linalg.norm(abs(f(evaluation) - lagrange(points, vals, evaluation)) /\
            lagrange(points, vals, evaluation))
        y = np.append(y, error_norm)
    
    plt.plot(x, y, label="Lagrange")
    y = np.array([])

    for n in range(4, 51):
        error_norm = np.linalg.norm(abs(f(evaluation) - splines(n, func, evaluation)) /\
            splines(n, func, evaluation))
        y = np.append(y, error_norm)

    plt.plot(x, y, label="splines")
    y = np.array([])

    for n in range(4, 51):
        points = chebyshev_nodes(n, func)
        vals = f(points)

        error_norm = np.linalg.norm(abs(f(evaluation) - lagrange(points, vals, evaluation)) /\
            lagrange(points, vals, evaluation))
        y = np.append(y, error_norm)
    
    plt.plot(x, y, label="Chebyshev")
    plt.yscale("log")
    plt.xlabel("Number of interpolation nodes")
    plt.ylabel("Value of error vector norm")
    plt.title(title)
    plt.legend()
    plt.show()

def zad2():
    func = 1
    error_plot(func)


zad1()
