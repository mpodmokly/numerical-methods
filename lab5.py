import numpy as np
import matplotlib.pyplot as plt

def approximation(years, vals, m):
    n = len(years)
    A = np.zeros((n, (m + 1)), dtype="float64")

    for i in range(n):
        for j in range(m + 1):
            A[i][j] = years[i] ** j

    c = np.linalg.inv(A.T @ A) @ A.T @ vals
    pol = np.polynomial.Polynomial(c)

    return pol
    

years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980],\
                  dtype="float64")
vals = np.array([76212168, 92228496, 106021537, 123202624, 132164569,\
                 151325798, 179323175, 203302031, 226542199], dtype="float64")
REAL90 = 248709873

for i in range(7):
    pol = approximation(years, vals, i)
    estimated = pol(1990)

    s = f"m={i} 1990: {round(estimated)}"
    s += f" err: {round(abs(REAL90 - estimated) / REAL90 * 100, 2)}%"
    print(s)


#plt.scatter(years, vals)
#plt.scatter(1990, pol(1990))
#x = np.linspace(1900, 1990, 1000)
#y = pol(x)

#plt.plot(x, y)
#plt.show()
