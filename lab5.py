import numpy as np
import matplotlib.pyplot as plt

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
        plt.scatter(1990, pol(1990), label="extrapolation to 1990")
        x = np.linspace(1900, 1990, 1000)
        y = pol(x)
        
        plt.plot(x, y)
        plt.xlabel("Year")
        plt.ylabel("Population")
        plt.title("US population approximation")
        plt.legend()
        plt.show()

def T0(x):
    return 1
def T1(x):
    return x
def T2(x):
    return 2 * x ** 2 - 1

def f(x):
    return np.sqrt(x)

def zad2():
    n = 2
    nodes = np.linspace(0, 2, 100)
    vals = f(nodes)
    c = np.polynomial.chebyshev.chebfit(nodes, vals, n)

    x = np.linspace(0, 2, 200)
    y = f(x)
    plt.plot(x, y, label="sqrt(x)")

    y = np.polynomial.chebyshev.Chebyshev(c)(x)
    plt.plot(x, y, label="approximation")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("f(x) = sqrt(x) polynomial approximation")
    plt.show()


#zad1(plot=True)
zad2()
