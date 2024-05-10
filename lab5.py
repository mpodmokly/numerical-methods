import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

YEARS = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980],\
                  dtype="float64")
POPULATION = np.array([76212168, 92228496, 106021537, 123202624, 132164569,\
                 151325798, 179323175, 203302031, 226542199], dtype="float64")
REAL_90 = 248709873

def approximation(m):
    n = len(YEARS)
    A = np.zeros((n, (m + 1)), dtype="float64")

    for i in range(n):
        for j in range(m + 1):
            A[i][j] = YEARS[i] ** j

    c = np.linalg.inv(A.T @ A) @ A.T @ POPULATION
    pol = np.polynomial.Polynomial(c)

    return pol

def AIC(m):
    n = len(YEARS)
    k = m + 1
    pol = approximation(m)

    aic_val = 2 * k + n * np.log(np.sum((POPULATION - pol(YEARS)) ** 2))
    aic_val += (2 * k * (k + 1)) / (n - k - 1)
    return aic_val

def zad1(plot = False):
    for i in range(7):
        pol = approximation(i)
        estimated = pol(1990)

        s = f"m={i} val: {round(estimated)}"
        s += f" err: {round(abs(REAL_90 - estimated) / REAL_90 * 100, 2)}%"
        print(s)

    for i in range(7):
        if i == 0:
            print()
        print(f"m={i} AIC = {round(AIC(i), 2)}")

    if plot:
        m = 2
        pol = approximation(m)
        plt.scatter(YEARS, POPULATION)
        plt.scatter(1990, pol(1990))
        x = np.linspace(1900, 1990, 1000)
        y = pol(x)
        
        plt.plot(x, y)
        plt.show()

def T0(x):
    return 1
def T1(x):
    return x
def T2(x):
    return 2 * x ** 2 - 1

def f(x):
    return np.sqrt(x)

def inner_product(f, g, w):
    integral, _ = quad(lambda x: f(x + 1) * g(x + 1) * w(x + 1), -1, 0)
    return integral

def weight_func(x):
    return 1 / np.sqrt(1 - x ** 2)

def coefficient(i):
    if i == 0:
        return inner_product(f, T0, weight_func) / inner_product(T0, T0, weight_func)#np.pi
    if i == 1:
        return inner_product(f, T1, weight_func) / inner_product(T1, T1, weight_func)#(np.pi / 2)
    if i == 2:
        return inner_product(f, T2, weight_func) / inner_product(T2, T2, weight_func)#(np.pi / 2)

def pol_val(c, x):
    return c[0] * T0(x - 1) + c[1] * T1(x - 1) + c[2] * T2(x - 1)


x = np.linspace(0, 2, 200)
y = np.sqrt(x)
plt.plot(x, y)

n = 2
c = np.array([coefficient(i) for i in range(n + 1)], dtype="float64")
#print(c)
#c = np.flip(c)
print(inner_product(T0, T0, weight_func))

y = pol_val(c, x)
#y = np.polynomial.chebyshev.chebfit()
plt.plot(x, y)
plt.show()

#zad1(plot=True)
