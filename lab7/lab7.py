import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad_vec
from scipy.integrate import quad

EVALS = 14

def f(x):
    return 4 / (1 + x ** 2)

eps = np.logspace(0, -EVALS, EVALS + 1)
x_trapz = np.array([])
y_trapz = np.array([])
x_gk = np.array([])
y_gk = np.array([])

for i in range(EVALS + 1):
    result, _, info = quad_vec(f, 0, 1, epsrel=eps[i], quadrature="trapezoid",\
                               full_output=True)
    x_trapz = np.append(x_trapz, info.neval)
    y_trapz = np.append(y_trapz, abs(result - np.pi) / np.pi)

    temp = quad(f, 0, 1, epsabs=eps[i], epsrel=eps[i],\
                           full_output=True)
    #x_gk = np.append(x_gk, info["neval"])
    #y_gk = np.append(y_gk, abs(result - np.pi) / np.pi)

#print(x_gk)
#print(y)

plt.plot(x_trapz, y_trapz)
#plt.yscale("log")
#plt.show()










# convergence = consistency + stability
