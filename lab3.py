import numpy as np
import matplotlib.pyplot as plt

def interpolation(years, vals, matrix, mode):
    coefficients = np.linalg.solve(matrix, vals)
    x = np.linspace(1900, 1980, 9)
    if mode == 1:
        y = np.polyval(coefficients, years)
    elif mode == 2:
        y = np.polyval(coefficients, years - 1900)
    elif mode == 3:
        y = np.polyval(coefficients, years - 1940)
    elif mode == 4:
        y = np.polyval(coefficients, (years - 1940) / 40)
    else:
        print("Invalid mode")
        return
    plt.scatter(x, y)

    x = np.linspace(1900, 1990, 91)
    if mode == 1:
        y = np.polyval(coefficients, x)
    elif mode == 2:
        y = np.polyval(coefficients, x - 1900)
    elif mode == 3:
        y = np.polyval(coefficients, x - 1940)
    elif mode == 4:
        y = np.polyval(coefficients, (x - 1940) / 40)
    plt.plot(x, y)

    real90 = 248709873
    if mode == 1:
        val90 = np.polyval(coefficients, 1990)
    elif mode == 2:
        val90 = np.polyval(coefficients, 1990 - 1900)
    elif mode == 3:
        val90 = np.polyval(coefficients, 1990 - 1940)
    elif mode == 4:
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

def newton_show(years, vals):
    x = np.linspace(1900, 1990, 10)
    y = newton(years, vals, x)
    plt.scatter(x, y)

    x = np.linspace(1900, 1990, 91)
    y = lagrange(years, vals, x)
    plt.plot(x, y)
    plt.show()

def interpolation_rounded(years, vals, matrix, mode):
    vals_rounded = np.round(vals, -6)
    coefficients_rounded = np.linalg.solve(matrix, vals_rounded)
    coefficients = np.linalg.solve(matrix, vals)

    print("\nnormal rounded")
    for i in range(n):
        print(coefficients[i], coefficients_rounded[i])
    
    x = np.linspace(1900, 1980, 9)
    if mode == 1:
        y = np.polyval(coefficients_rounded, years)
    elif mode == 2:
        y = np.polyval(coefficients_rounded, years - 1900)
    elif mode == 3:
        y = np.polyval(coefficients_rounded, years - 1940)
    elif mode == 4:
        y = np.polyval(coefficients_rounded, (years - 1940) / 40)
    else:
        print("Invalid mode")
        return
    plt.scatter(x, y)

    x = np.linspace(1900, 1990, 91)
    if mode == 1:
        y = np.polyval(coefficients_rounded, x)
    elif mode == 2:
        y = np.polyval(coefficients_rounded, x - 1900)
    elif mode == 3:
        y = np.polyval(coefficients_rounded, x - 1940)
    elif mode == 4:
        y = np.polyval(coefficients_rounded, (x - 1940) / 40)
    plt.plot(x, y)

    real90 = 248709873
    if mode == 1:
        val90 = np.polyval(coefficients_rounded, 1990)
    elif mode == 2:
        val90 = np.polyval(coefficients_rounded, 1990 - 1900)
    elif mode == 3:
        val90 = np.polyval(coefficients_rounded, 1990 - 1940)
    elif mode == 4:
        val90 = np.polyval(coefficients_rounded, (1990 - 1940) / 40)
    print(f"year 1990: {round(val90)}")
    print(f"relative error: {round(abs(real90 - val90) / real90 * 100, 2)}%")

    plt.scatter(1990, real90, color="green")
    plt.scatter(1990, val90, color="red")
    plt.show()


years = np.float64(np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980]))
vals = np.float64(np.array([76212168, 92228496, 106021537, 123202624, 132164569,
                            151325798, 179323175, 203302031, 226542199]))

n = len(years)
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

#interpolation(years, vals, vand4, 4)
#lagrange_show(years, vals)
#newton_show(years, vals)
interpolation_rounded(years, vals, vand4, 4)
