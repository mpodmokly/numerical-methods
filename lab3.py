import numpy as np
import matplotlib.pyplot as plt

REAL90 = 248709873

def interpolation(years, vals):
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

    coefficients = np.linalg.solve(vand4, vals)
    return coefficients

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

def divided_differences(x, y):
    n = len(x)
    matrix = [[] for _ in range(n)]
    
    for i in range(n):
        matrix[i].append(y[i])
    
    for i in range(1, n):
        for j in range(1, i + 1):
            matrix[i].append((matrix[i][j - 1] - matrix[i - 1][j - 1]) / (x[i] - x[i - j]))
    
    coefficients = []
    for i in range(n):
        coefficients.append(matrix[i][-1])
    
    return coefficients

def newton(years, vals, x):
    n = len(years)
    coefficients = divided_differences(years, vals)
    term = [1]

    for i in range(1, n):
        term.append(term[i - 1] * (x - years[i - 1]))
    
    for i in range(n):
        term[i] *= coefficients[i]
    
    result = sum(term)
    return result

def interpolation_rounded(years, vals):
    n = len(years)
    vals_rounded = np.round(vals, -6)
    vand4 = np.vander((years - 1940) / 40)

    coefficients_rounded = np.linalg.solve(vand4, vals_rounded)
    coefficients = np.linalg.solve(vand4, vals)

    print("\nnormal | rounded")
    for i in range(n):
        print(coefficients[i], coefficients_rounded[i])
    
    return coefficients_rounded


years = np.float64(np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980]))
vals = np.float64(np.array([76212168, 92228496, 106021537, 123202624, 132164569,
                            151325798, 179323175, 203302031, 226542199]))


coefficients = interpolation(years, vals)
coefficients_rounded = interpolation_rounded(years, vals)
pol = np.polynomial.Polynomial(np.flip(coefficients))
pol_rounded = np.polynomial.Polynomial(np.flip(coefficients_rounded))

x = np.linspace(1900, 1990, 10)
y = pol((x - 1940) / 40)
plt.scatter(x, y)

x = np.linspace(1900, 1990, 91)
y = pol((x - 1940) / 40)
plt.plot(x, y, label="unrounded data")

x = np.linspace(1900, 1990, 10)
y = pol_rounded((x - 1940) / 40)
plt.scatter(x, y)

x = np.linspace(1900, 1990, 91)
y = pol_rounded((x - 1940) / 40)
plt.plot(x, y, label="data rounded to the nearest million")
plt.scatter(1990, REAL90, color="green", label="real population in 1990")

val90 = pol((1990 - 1940) / 40)
print("\nnot rounded:")
print(f"year 1990: {round(val90)}")
print(f"relative error: {round(abs(REAL90 - val90) / REAL90 * 100, 2)}%")

val90_rounded = pol_rounded((1990 - 1940) / 40)
print("\nrounded:")
print(f"year 1990: {round(val90_rounded)}")
print(f"relative error: {round(abs(REAL90 - val90_rounded) / REAL90 * 100, 2)}%")

plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Interpolation of the US population data in years 1900-1990")
plt.legend()
plt.show()
