import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad_vec

def f(x):
    return 4 / (1 + x ** 2)


eps = np.logspace(0, -14, 15)
x = np.array([])
y = np.array([])

for i in range(15):
    result, _, info = quad_vec(f, 0, 1, epsabs=eps[i], epsrel=eps[i],\
                               quadrature="gk15", full_output=True)
    x = np.append(x, info.neval)
    y = np.append(y, abs(result - np.pi) / np.pi)

print(x)
#print(y)

#plt.plot(x, y)
#plt.yscale("log")
#plt.show()










# convergence = consistency + stability
