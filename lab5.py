import numpy as np

years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980],\
                  dtype="float64")
vals = np.array([76212168, 92228496, 106021537, 123202624, 132164569,
                 151325798, 179323175, 203302031, 226542199], dtype="float64")


n = len(years)
m = 3
A = np.zeros((n, m), dtype="float64")

for i in range(n):
    for j in range(m):
        A[i][j] = years[i] ** j

print(A)
