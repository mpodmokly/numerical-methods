import numpy as np

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
    iter = 10
    eps = 1e-13

    x = np.zeros(shape=(n+1, 2))
    x[:, 0] = np.random.uniform(0, 20, n+1)
    x[:, 1] = np.random.uniform(0, 20, n+1)
    x_0 = np.array([0, 0])
    x_n = np.array([20, 20])
    x[0] = x_0
    x[n] = x_n

    def F(x):
        result = 0

        for i in range(n + 1):
            for j in range(k):
                result += 1 / (eps + np.linalg.norm(x[i] - r[j])) ** 2
        result *= l1

        for i in range(n):
            result += np.linalg.norm(x[i + 1] - x[i]) ** 2
        result *= l2
        return result

    def gradient(x):
        grad = np.zeros_like(x)
        grad[0] = 2 * l2 * (x[1] - x[0]) - l1 *\
        np.sum((2 * (x[0] - r)) / (eps + np.linalg.norm(x[0] - r)))
        
        
        for i in range(1, n):
            grad[i] = 2 * l2 * (x[i + 1] - x[i - 1]) - l1 *\
                np.sum((2 * (x[i] - r)) / (eps + np.linalg.norm(x[i] - r)))
        
        grad[-1] = 2 * l2 * (x[n] - x[n - 1]) - l1 *\
            np.sum((2 * (x[n] - r)) / (eps + np.linalg.norm(x[n] - r)))
        
        return grad 

    def gss(f, a, b):
        tolerance = 1e-5
        invphi = (np.sqrt(5) - 1) / 2

        while abs(b - a) > tolerance:
            c = b - (b - a) * invphi
            d = a + (b - a) * invphi
            if f(c) < f(d):
                b = d
            else:
                a = c

        return (b + a) / 2
    
    def sdm(x0):
        for _ in range(iter):
            d = gradient(x0)
            #print(d)
            alpha = gss(lambda a: F(x0 - a * d), 0, 1)
            x0 = x0 - alpha * d
        
        print(x0)
        return x0
    
    sdm(x)


zad2()
