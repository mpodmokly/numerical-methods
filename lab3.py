import numpy as np
import matplotlib.pyplot as plt

years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
vals = np.array([76212168, 92228496, 106021537, 123202624, 132164569,
                    151325798, 179323175, 203302031, 226542199])

plt.scatter(years, vals, color="green")

vand1 = np.vander(years)
vand2 = np.vander(np.float64(years - 1900))
vand3 = np.vander(np.float64(years - 1940))
vand4 = np.vander((years - 1940) / 40)

cond1 = np.linalg.cond(vand1)
cond2 = np.linalg.cond(vand2)
cond3 = np.linalg.cond(vand3)
cond4 = np.linalg.cond(vand4)

print(f"Cond 1: {cond1}")
print(f"Cond 2: {cond2}")
print(f"Cond 3: {cond3}")
print(f"Cond 4: {cond4} (best)")

coefficients = np.linalg.solve(vand2, vals)
x = np.linspace(1900, 1980, 9)
y = np.polyval(coefficients, np.float64(years - 1900))
plt.scatter(x, y, color="red")

x = np.linspace(1900, 1980, 1000)
y = np.polyval(coefficients, np.float64(x - 1900))
plt.plot(x, y)
plt.show()
