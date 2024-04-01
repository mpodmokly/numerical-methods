import numpy as np
import matplotlib.pyplot as plt

def interpolation(years, vals, matrix):
    coefficients = np.linalg.solve(matrix, vals)
    x = np.linspace(1900, 1980, 9)
    y = np.polyval(coefficients, (years - 1940) / 40)
    plt.scatter(x, y)

    x = np.linspace(1900, 1990, 91)
    y = np.polyval(coefficients, (x - 1940) / 40)
    plt.plot(x, y)

    real90 = 248709873
    val90 = np.polyval(coefficients, (1990 - 1940) / 40)
    print(f"year 1990: {round(val90)}")
    print(f"relative error: {round(abs(real90 - val90) / real90 * 100, 2)}%")

    plt.scatter(1990, real90, color="green")
    plt.scatter(1990, val90, color="red")
    plt.show()

def lagrange(years, vals, x):
    n = len(years)
    result = 0

    for i in range(n):
        product = 1
        for j in range(n):
            if j != i:
                product *= (x - years[j]) / (years[i] - years[j])
        result += product * vals[i]
    
    return result

def lagrange_show(years, vals):
    x = np.linspace(1900, 1990, 10)
    y = lagrange(years, vals, x)
    plt.scatter(x, y)

    x = np.linspace(1900, 1990, 91)
    y = lagrange(years, vals, x)
    plt.plot(x, y)
    plt.show()

def newton(years, vals, x):
    n = len(years)


years = np.float64(np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980]))
vals = np.float64(np.array([76212168, 92228496, 106021537, 123202624, 132164569,
                            151325798, 179323175, 203302031, 226542199]))

vand1 = np.vander(years)
vand2 = np.vander(years - 1900)
vand3 = np.vander(years - 1940)
vand4 = np.vander((years - 1940) / 40)

cond1 = np.linalg.cond(vand1)
cond2 = np.linalg.cond(vand2)
cond3 = np.linalg.cond(vand3)
cond4 = np.linalg.cond(vand4)

print(f"Cond 1: {cond1:.2e}")
print(f"Cond 2: {cond2:.2e}")
print(f"Cond 3: {cond3:.2e}")
print(f"Cond 4: {cond4:.2e} (best)")

#interpolation(years, vals, vand4)
lagrange_show(years, vals)
