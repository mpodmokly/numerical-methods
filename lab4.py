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

def f1(x):
    return 1 / (1 + 25 * x * x)

def f2(x):
    return np.exp(np.cos(x))


x = np.linspace(-1, 1, 1000)
y = f1(x)
plt.plot(x, y, label="Runge's function")

n = 12
points = np.linspace(-1, 1, n)
vals = f1(points)
plt.scatter(points, vals, color="orange")

y = lagrange(points, vals, x)
plt.plot(x, y, color="orange", label="Lagrange polynomial")

for i in range(n - 1):
    interval_points = np.linspace(points[i], points[i + 1], 4)
    interval_vals = f1(interval_points)

    interval_x = np.linspace(points[i], points[i + 1], 100)
    interval_y = lagrange(interval_points, interval_vals, interval_x)

    if i == 0:
        plt.plot(interval_x, interval_y, color="red", label="splines")
    else:
        plt.plot(interval_x, interval_y, color="red")

chebyshev = np.array([])
for i in range(n + 1):
    chebyshev = np.append(chebyshev, np.cos((2 * i + 1) / (2 * (n + 1)) * np.pi))

che_vals = f1(chebyshev)
plt.scatter(chebyshev, che_vals, color="green")
y = lagrange(chebyshev, che_vals, x)
plt.plot(x, y, color="green", label="Lagrange polynomial using Chebyshev nodes")


plt.xlabel("Arguments")
plt.ylabel("Values")
plt.title("Runge's function interpolation using various methods")
plt.legend()
plt.show()
