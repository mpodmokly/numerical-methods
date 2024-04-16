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
    a = x // (2 / (n - 1)) * (2 / (n - 1))
    b = a + (2 / (n - 1))

    points = np.linspace(a, b, 4)
    vals = func(points)

    return lagrange(points, vals, x)

def chebyshev_nodes(n):
    chebyshev = np.array([])
    for i in range(n + 1):
        chebyshev = np.append(chebyshev, np.cos((2 * i + 1) / (2 * (n + 1)) * np.pi))
    
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

    y = splines(n, x)
    plt.plot(x, y, color="red", label="splines")

    chebyshev = chebyshev_nodes(n)
    che_vals = f1(chebyshev)

    plt.scatter(chebyshev, che_vals, color="green")
    y = lagrange(chebyshev, che_vals, x)
    plt.plot(x, y, color="green", label="Chebyshev")

    plt.xlabel("Arguments")
    plt.ylabel("Values")
    plt.title("Runge's function interpolation")
    plt.legend()
    plt.show()

def error_plot(func, title):
    evaluation = np.random.uniform(-1, 1, 500)
    x = np.linspace(4, 50, 47)
    y = np.array([])

    for n in range(4, 51):
        points = np.linspace(-1, 1, n)
        vals = func(points)

        error_norm = np.linalg.norm(abs(func(evaluation) - lagrange(points, vals, evaluation)) /\
            lagrange(points, vals, evaluation))
        y = np.append(y, error_norm)
    
    plt.plot(x, y, label="Lagrange")
    y = np.array([])

    for n in range(4, 51):
        error_norm = np.linalg.norm(abs(func(evaluation) - splines(n, func, evaluation)) /\
            splines(n, func, evaluation))
        y = np.append(y, error_norm)

    plt.plot(x, y, label="splines")
    y = np.array([])

    for n in range(4, 51):
        points = chebyshev_nodes(n)
        vals = func(points)

        error_norm = np.linalg.norm(abs(func(evaluation) - lagrange(points, vals, evaluation)) /\
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
    #error_plot(f1, "Error comparison for Runge's function")
    error_plot(f2, "Error comparison for exponentional function")


zad2()
