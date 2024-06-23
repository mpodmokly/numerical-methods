import numpy as np
import matplotlib.pyplot as plt

def f1(x, y):
    return x ** 2 - 4 * x * y + y ** 2
def f2(x, y):
    return x ** 4 - 4 * x * y + y ** 4
def f3(x, y):
    return 2 * x ** 3 - 3 * x ** 2 - 6 * x * y * (x - y - 1)
def f4(x, y):
    return (x - y) ** 4 + x ** 2 - y ** 2 - 2 * x + 2 * y + 1

def grad_f1(x, y):
    df_dx = 2 * x - 4 * y
    df_dy = 2 * y - 4 * x
    return np.array([df_dx, df_dy]).reshape(-1, 1)
def hessian_f1(x, y):
    df_xx = 2
    df_xy = -4
    df_yx = -4
    df_yy = 2
    return np.array([[df_xx, df_xy], [df_yx, df_yy]])

def newton(f, grad, hessian, x0):
    x0 = np.array(x0).reshape(-1, 1)

    k = 2
    for _ in range(k):
        x0 = x0 - np.linalg.inv(hessian(*x0.reshape(-1))) @ grad(*x0.reshape(-1))

    print(x0)
    return x0


x0 = (1, 1)
#newton(f1, grad_f1, hessian_f1, x0)

eig, _ = np.linalg.eig(hessian_f1(0, 0))
print(eig)
