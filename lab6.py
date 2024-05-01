import numpy as np

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
    
    print(integral)
    return integral

m = 5
simpson(m)
