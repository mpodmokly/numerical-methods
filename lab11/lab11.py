import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f1(x, y):
    return x ** 2 - 4 * x * y + y ** 2

def f2(x, y):
    return x ** 4 - 4 * x * y + y ** 4

def f3(x, y):
    return 2 * x ** 3 - 3 * x ** 2 - 6 * x * y * (x - y - 1)

def f4(x, y):
    return (x - y) ** 4 + x ** 2 - y ** 2 - 2 * x + 2 * y + 1


x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)
Z = f4(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")

plt.show()
