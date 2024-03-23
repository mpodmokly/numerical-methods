import numpy as np
import matplotlib.pyplot as plt

def f1(t, j):
    return t ** (j - 1)

def f2(t, j):
    return (t - 1900) ** (j - 1)

def f3(t, j):
    return (t - 1940) ** (j - 1)

def f4(t, j):
    return ((t - 1940) / 40) ** (j - 1)

def polynomial(n, coefficients, x):
    value = 0
    for i in range(n):
        value += coefficients[i] * x ** i
    return value

years = np.array([1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980])
vals = [76212168, 92228496, 106021537, 123202624, 132164569,
        151325798, 179323175, 203302031, 226542199]

vand1 = np.array([[float(f1(t, j)) for j in range(1, 10)] for t in years])
vand2 = np.array([[float(f2(t, j)) for j in range(1, 10)] for t in years])
vand3 = np.array([[float(f3(t, j)) for j in range(1, 10)] for t in years])
vand4 = np.array([[float(f4(t, j)) for j in range(1, 10)] for t in years])
cond1 = np.linalg.cond(vand1)
cond2 = np.linalg.cond(vand2)
cond3 = np.linalg.cond(vand3)
cond4 = np.linalg.cond(vand4)

print(f"Cond 1: {cond1}")
print(f"Cond 2: {cond2}")
print(f"Cond 3: {cond3}")
print(f"Cond 4: {cond4} (best)")

a = np.linalg.solve(vand1, vals)
x = np.linspace(1900, 1990, 91)
y = polynomial(len(a), a, x)

plt.plot(x, y)

x = np.linspace(1900, 1980, 9)
y = polynomial(len(a), a, x)
plt.scatter(x, y)
#plt.show()

val90 = polynomial(len(a), a, 1990)
print(f"1990: {val90}")
true90 = 248709873
print(f"true 1990: {true90}")
print(f"relative error: {abs(val90 - true90) / true90 * 100}%")
