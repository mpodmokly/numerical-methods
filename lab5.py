import numpy as np
import matplotlib.pyplot as plt

years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980],\
                  dtype="float64")
vals = np.array([76212168, 92228496, 106021537, 123202624, 132164569,
                 151325798, 179323175, 203302031, 226542199], dtype="float64")

plt.scatter(years, vals)

n = len(years)
m = 4
A = np.zeros((n, (m + 1)), dtype="float64")

for i in range(n):
    for j in range(m + 1):
        A[i][j] = years[i] ** j


c = np.linalg.inv(A.T @ A) @ A.T @ vals
print(A)
print(c)

pol = np.polynomial.Polynomial(c)

x = np.linspace(1900, 1980, 1000)
y = pol(x)
plt.plot(x, y)
plt.show()
