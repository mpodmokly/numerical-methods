import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# convergence = consistency + stability

def zad2():
    def f(y, t):
        return -5 * y

    y0 = 1
    h = 0.1
    t_end = 0.5

    steps_no = int(t_end / h)

    t_vals = np.linspace(0, t_end, steps_no + 1)
    y_vals = np.zeros(steps_no + 1)
    y_implicit = np.zeros(steps_no + 1)
    y_vals[0] = y0
    y_implicit[0] = y0

    for i in range(steps_no):
        y_vals[i + 1] = y_vals[i] + h * f(y_vals[i], t_vals[i])
        y_implicit[i + 1] = y_implicit[i] / (5 * h + 1)

    euler_val = y_vals[-1]
    implicit_val = y_implicit[-1]
    true_val = y0 * np.exp(-5 * t_end)
    print(f"Euler's explicit method: {euler_val}")
    print(f"Euler's implicit method: {implicit_val}")
    print(f"True value: {true_val}")

def zad3(plot_N = False, method = 1):
    N = 763
    BETA = 1
    GAMMA = 1 / 7

    def f1(S, I, R, t):
        return - (BETA / N) * I * S
    def f2(S, I, R, t):
        return (BETA / N) * I * S - GAMMA * I
    def f3(S, I, R, t):
        return GAMMA * I

    def euler_explicit(Svals, Ivals, Rvals, t_vals, h, steps):
        for i in range(steps):
            Svals[i + 1] = Svals[i] + h * f1(Svals[i], Ivals[i],\
                                            Rvals[i], t_vals[i])
            Ivals[i + 1] = Ivals[i] + h * f2(Svals[i], Ivals[i],\
                                            Rvals[i], t_vals[i])
            Rvals[i + 1] = Rvals[i] + h * f3(Svals[i], Ivals[i],\
                                            Rvals[i], t_vals[i])
        return Svals, Ivals, Rvals
    
    def euler_step(S_prev, I_prev, R_prev, h):
        def equations(vars):
            S_new, I_new, R_new = vars
            eq1 = S_new - S_prev - h * f1(S_new, I_new, R_new, 0)
            eq2 = I_new - I_prev - h * f2(S_new, I_new, R_new, 0)
            eq3 = R_new - R_prev - h * f3(S_new, I_new, R_new, 0)
            return [eq1, eq2, eq3]
        
        return fsolve(equations, [S_prev, I_prev, R_prev])

    def euler_implicit(Svals, Ivals, Rvals, t_vals, h, steps):
        for i in range(steps):
            S_new, I_new, R_new = euler_step(Svals[i], Ivals[i], Rvals[i], h)
            Svals[i + 1] = S_new
            Ivals[i + 1] = I_new
            Rvals[i + 1] = R_new
        return Svals, Ivals, Rvals
    
    def RK4(Svals, Ivals, Rvals, t_vals, h, steps):
        for i in range(steps):
            k1 = f1(Svals[i], Ivals[i], Rvals[i], t_vals[i])
            k2 = f1(Svals[i] + h * k1 / 2, Ivals[i] + h * k1 / 2, Rvals[i] + h * k1 / 2, 0)
            k3 = f1(Svals[i] + h * k2 / 2, Ivals[i] + h * k2 / 2, Rvals[i] + h * k2 / 2, 0)
            k4 = f1(Svals[i] + h * k3, Ivals[i] + h * k3, Rvals[i] + h * k3, 0)
            Svals[i + 1] = Svals[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            
            k1 = f2(Svals[i], Ivals[i], Rvals[i], t_vals[i])
            k2 = f2(Svals[i] + h * k1 / 2, Ivals[i] + h * k1 / 2, Rvals[i] + h * k1 / 2, 0)
            k3 = f2(Svals[i] + h * k2 / 2, Ivals[i] + h * k2 / 2, Rvals[i] + h * k2 / 2, 0)
            k4 = f2(Svals[i] + h * k3, Ivals[i] + h * k3, Rvals[i] + h * k3, 0)
            Ivals[i + 1] = Ivals[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

            k1 = f3(Svals[i], Ivals[i], Rvals[i], t_vals[i])
            k2 = f3(Svals[i] + h * k1 / 2, Ivals[i] + h * k1 / 2, Rvals[i] + h * k1 / 2, 0)
            k3 = f3(Svals[i] + h * k2 / 2, Ivals[i] + h * k2 / 2, Rvals[i] + h * k2 / 2, 0)
            k4 = f3(Svals[i] + h * k3, Ivals[i] + h * k3, Rvals[i] + h * k3, 0)
            Rvals[i + 1] = Rvals[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return Svals, Ivals, Rvals

    t_start = 0
    t_end = 14
    h = 0.2
    steps = int((t_end - t_start) / h)

    t_vals = np.linspace(t_start, t_end, steps + 1)
    Svals = np.zeros(steps + 1)
    Ivals = np.zeros(steps + 1)
    Rvals = np.zeros(steps + 1)
    Svals[0] = 762
    Ivals[0] = 1
    Rvals[0] = 0

    if plot_N:
        Svals, Ivals, Rvals = euler_explicit(Svals, Ivals, Rvals, t_vals, h, steps)
        plt.plot(t_vals, Svals + Ivals + Rvals, label="explicit Euler")
        Svals, Ivals, Rvals = euler_implicit(Svals, Ivals, Rvals, t_vals, h, steps)
        plt.plot(t_vals, Svals + Ivals + Rvals, label="implicit Euler")
        Svals, Ivals, Rvals = RK4(Svals, Ivals, Rvals, t_vals, h, steps)
        plt.plot(t_vals, Svals + Ivals + Rvals, label="RK4")
        plt.xlabel("Days")
        plt.ylabel("People")
        plt.title("Total number of people")
        plt.legend()
        plt.show()
    else:
        if method == 1:
            Svals, Ivals, Rvals = euler_explicit(Svals, Ivals, Rvals, t_vals, h, steps)
            title = " explicit Euler method"
        elif method == 2:
            Svals, Ivals, Rvals = euler_implicit(Svals, Ivals, Rvals, t_vals, h, steps)
            title = " implicit Euler method"
        else:
            Svals, Ivals, Rvals = RK4(Svals, Ivals, Rvals, t_vals, h, steps)
            title = " RK4 method"
    
        plt.plot(t_vals, Svals, label="susceptible")
        plt.plot(t_vals, Ivals, label="infectious")
        plt.plot(t_vals, Rvals, label="recovered")
        plt.xlabel("Days")
        plt.ylabel("People")
        plt.title("Course of the epidemic in the population" + title)
        plt.legend()
        plt.show()

def zad3_mincost():
    N = 500
    DAYS = 14
    true_I = np.array([1, 3, 6, 25, 73, 222, 294, 258, 237, 191,\
                       125, 69, 27, 11, 4])
    
    def f1(S, I, beta, gamma):
        return -(beta / N) * I * S
    def f2(S, I, beta, gamma):
        return (beta / N) * I * S - gamma * I

    def euler_explicit_I(Svals, Ivals, h, steps, beta, gamma):
        for i in range(steps):
            Svals[i + 1] = Svals[i] + h * f1(Svals[i], Ivals[i], beta, gamma)
            Ivals[i + 1] = Ivals[i] + h * f2(Svals[i], Ivals[i], beta, gamma)
        return Ivals

    
    def cost_function_1(theta):
        beta, gamma = theta
        h = 1
        steps = int(DAYS / h)

        Ivals = np.zeros(steps + 1)
        Svals = np.zeros(steps + 1)
        Svals[0] = N - 1
        Ivals[0] = 1

        Ivals = euler_explicit_I(Svals, Ivals, h, steps, beta, gamma)
        return np.sum((true_I - Ivals) ** 2)
    
    def cost_function_2(theta):
        beta, gamma = theta
        h = 1
        steps = int(DAYS / h)

        Ivals = np.zeros(steps + 1)
        Svals = np.zeros(steps + 1)
        Svals[0] = N - 1
        Ivals[0] = 1

        Ivals = euler_explicit_I(Svals, Ivals, h, steps, beta, gamma)
        return -np.sum(true_I * np.log(Ivals + 1e-9)) + np.sum(Ivals)
    
    initial_guess = np.array([0.1, 0.1])
    result = minimize(cost_function_2, initial_guess, method="Nelder-Mead")
    print(result["x"])
    print(result["x"][0] / result["x"][1])


#method = 3
#plot_N = True
#zad3(plot_N, method)
zad3_mincost()
