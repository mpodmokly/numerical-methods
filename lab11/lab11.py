import numpy as np
import matplotlib.pyplot as plt

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
def grad_f2(x, y):
    df_dx = 4 * x ** 3 - 4 * y
    df_dy = 4 * y ** 3 - 4 * x
    return np.array([df_dx, df_dy]).reshape(-1, 1)
def hessian_f2(x, y):
    df_xx = 12 * x ** 2
    df_xy = -4
    df_yx = -4
    df_yy = 12 * y ** 2
    return np.array([[df_xx, df_xy], [df_yx, df_yy]])
def grad_f3(x, y):
    df_dx = 6 * x ** 2 - 12 * x * y - 6 * x + 6 * y ** 2 + 6 * y
    df_dy = -6 * x ** 2 + 12 * x * y + 6 * x
    return np.array([df_dx, df_dy]).reshape(-1, 1)
def hessian_f3(x, y):
    df_xx = 12 * x - 12 * y - 6
    df_xy = -12 * x + 12 * y + 6
    df_yx = -12 * x + 12 * y + 6
    df_yy = 12 * x
    return np.array([[df_xx, df_xy], [df_yx, df_yy]])
def grad_f4(x, y):
    df_dx = 4 * x ** 3 - 12 * x ** 2 * y + 12 * x * y ** 2 + 2 * x - 4 * y ** 3 - 2
    df_dy = -4 * x ** 3 + 12 * x ** 2 * y - 12 * x * y ** 2 + 4 * y ** 3 - 2 * y + 2
    return np.array([df_dx, df_dy]).reshape(-1, 1)
def hessian_f4(x, y):
    df_xx = 12 * x ** 2 - 24 * x * y + 12 * y ** 2 + 2
    df_xy = -12 * x ** 2 + 24 * x * y - 12 * y ** 2 
    df_yx = -12 * x ** 2 + 24 * x * y  - 12 * y ** 2
    df_yy = 12 * x ** 2 - 24 * x * y + 12 * y ** 2 - 2
    return np.array([[df_xx, df_xy], [df_yx, df_yy]])

def newton(grad, hessian, x0, k):
    x0 = np.array(x0).reshape(-1, 1)

    for _ in range(k):
        x0 = x0 - np.linalg.inv(hessian(*x0.reshape(-1))) @ grad(*x0.reshape(-1))
    
    return x0

def classify_point(grad, hessian, x0):
    k = 7
    p = newton(grad, hessian, x0, k)
    eig, _ = np.linalg.eig(hessian(*p.reshape(-1)))

    if np.all(eig > 0):
        print(p.reshape(-1), "- minimum")
    elif np.all(eig < 0):
        print(p.reshape(-1), "- maximum")
    else:
        print(p.reshape(-1), "- saddle point")

def zad1():
    print("f1:")
    x0 = (1, 1)
    classify_point(grad_f1, hessian_f1, x0)

    print("f2:")
    x0 = (-2, -2)
    classify_point(grad_f2, hessian_f2, x0)
    x0 = (2, 2)
    classify_point(grad_f2, hessian_f2, x0)
    x0 = (-0.5, 0)
    classify_point(grad_f2, hessian_f2, x0)

    print("f3:")
    x0 = (-2, -2)
    classify_point(grad_f3, hessian_f3, x0)
    x0 = (2, 0)
    classify_point(grad_f3, hessian_f3, x0)
    x0 = (0, -2)
    classify_point(grad_f3, hessian_f3, x0)
    x0 = (0, 1)
    classify_point(grad_f3, hessian_f3, x0)

    print("f4:")
    x0 = (0, 0)
    classify_point(grad_f4, hessian_f4, x0)


def zad2():
    n = 20
    k = 50
    r = np.random.uniform(0, 20, size=(k, 2))
    l1 = 1
    l2 = 1
    iter = 400
    eps = 1e-13

    x = np.zeros(shape=(n + 1, 2))
    x[:, 0] = np.random.uniform(0, 20, n + 1)
    x[:, 1] = np.random.uniform(0, 20, n + 1)
    x_0 = np.array([0, 0])
    x_n = np.array([20, 20])
    x[0] = x_0
    x[n] = x_n

    def F(x):
        result = 0

        for i in range(n + 1):
            for j in range(k):
                result += l1 / (eps + np.linalg.norm(x[i] - r[j])) ** 2

        for i in range(n):
            result += l2 * np.linalg.norm(x[i + 1] - x[i]) ** 2
        
        return result

    def gradient(x):
        grad = np.zeros_like(x)

        for i in range(1, n):   
            grad[i] += 2 * l2 * (2 * x[i] - x[i - 1] - x[i + 1])

            for j in range(k):
                norm = np.linalg.norm(x[i] - r[j])
                grad[i] -= 2 * l1 * (x[i] - r[j]) / (eps + norm ** 2) ** 2

        return grad

    def gss(f, a, b):
        tolerance = 1e-5
        invphi = (np.sqrt(5) - 1) / 2

        while abs(b - a) > tolerance:
            alpha1 = b - (b - a) * invphi
            alpha2 = a + (b - a) * invphi
            if f(alpha1) < f(alpha2):
                b = alpha2
            else:
                a = alpha1

        return (a + b) / 2
    
    def sdm(x0):
        F_values = np.zeros(iter)

        for i in range(iter):
            d = gradient(x0)
            alpha = gss(lambda a: F(x0 - a * d), 0, 1)
            x0 = x0 - alpha * d
            F_values[i] = F(x0)
        
        return x0, F_values
    
    def plot_path(x, r):
        plt.scatter(r[:, 0], r[:, 1], color="red", label="r")
        plt.scatter(x[:, 0], x[:, 1], color="green", label="x")
        plt.plot(x[:, 0], x[:, 1], color="green")
        plt.grid(True)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("The shortest path")
        plt.legend()
        plt.show()
    
    def plot_F(F_values):
        x = np.linspace(1, iter, iter)
        plt.plot(x, F_values)
        plt.xlabel("iteration")
        plt.ylabel("F value")
        plt.title("Values of the F function")
        plt.show()
    
    x, F_values = sdm(x)
    plot_path(x, r)
    #plot_F(F_values)


zad2()
