import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

N_TRAIN = 200
N_TEST = 1000
OMEGA = 1
LR = 0.001
EPOCHS = 10000

def loss_fun(model, x_train):
    x_train.requires_grad_(True)

    u_pred = model(x_train)
    du_dx = torch.autograd.grad(outputs=u_pred, inputs=x_train,
                                grad_outputs=torch.ones_like(u_pred),
                                create_graph=True)[0]
    target = torch.cos(OMEGA * x_train)
    residual_loss = nn.functional.mse_loss(du_dx, target)

    boundary = model(torch.tensor([0], dtype=torch.float32))
    boundary_loss = torch.norm(boundary) ** 2
    return residual_loss + boundary_loss

def neural_network(layers):
    n = len(layers)
    modules = []

    for i in range(n - 1):
        modules.append(nn.Linear(layers[i], layers[i + 1]))
        if i < n - 2:
            modules.append(nn.Tanh())
    
    return nn.Sequential(*modules)


layers = [1] + 2 * [16] + [1]
model = neural_network(layers)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

a = -2 * np.pi
b = 2 * np.pi

x = np.linspace(a, b, N_TRAIN - 1)
x = np.append(x, 0)
x = x.reshape(-1, 1)
x_train = torch.tensor(x, dtype=torch.float32)

for i in range(EPOCHS):
    optimizer.zero_grad()
    loss = loss_fun(model, x_train)
    loss.backward()
    optimizer.step()

    if i % 100 == 99:
        print(f"({np.round((i + 1) / EPOCHS * 100, 1)}%) {np.round(loss.item(), 4)}")

x = np.linspace(a, b, N_TEST + 2)
x = np.delete(x, 0)
x = np.delete(x, -1)
x = x.reshape(-1, 1)
x_test = torch.tensor(x, dtype=torch.float32)

prediction = model(x_test).detach().numpy()
exact_result = (np.sin(OMEGA * x_test.numpy()) / OMEGA).astype(np.float32)

plt.plot(x_test.numpy(), prediction, label="PINN prediction")
plt.plot(x_test.numpy(), exact_result, label="Exact solution")
plt.legend()
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Comparison neural network predictions and real solution")
plt.show()
