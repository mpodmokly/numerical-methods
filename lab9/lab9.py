import numpy as np
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

def zad3():
    N = 763
    BETA = 1
    GAMMA = 1 / 7

    def f1(S, I, R, t):
        return - (BETA / N) * I * S
    def f2(S, I, R, t):
        return (BETA / N) * I * S - GAMMA * I
    def f3(S, I, R, t):
        return GAMMA * I

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

    for i in range(steps):
        Svals[i + 1] = Svals[i] + h * f1(Svals[i], Ivals[i],\
                                         Rvals[i], t_vals[i])
        Ivals[i + 1] = Ivals[i] + h * f2(Svals[i], Ivals[i],\
                                         Rvals[i], t_vals[i])
        Rvals[i + 1] = Rvals[i] + h * f3(Svals[i], Ivals[i],\
                                         Rvals[i], t_vals[i])
    
    plt.plot(t_vals, Svals, label="susceptible")
    plt.plot(t_vals, Ivals, label="infectious")
    plt.plot(t_vals, Rvals, label="recovered")
    plt.xlabel("Days")
    plt.ylabel("People")
    plt.title("Course of the epidemic in the population")
    plt.legend()
    plt.show()


zad3()
