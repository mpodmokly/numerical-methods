import torch
import torch.nn as nn
import numpy as np

N = 10
OMEGA = 1
LR = 0.001
EPOCHS = 100

def loss_fun(model, x_train):
    target = torch.cos(OMEGA * x_train)
    residual_loss = nn.functional.mse_loss(model(x_train), target)

    boundary = model(torch.tensor([0], dtype=torch.float32))
    boundary_loss = torch.norm(boundary) ** 2

    loss = residual_loss + boundary_loss
    return loss

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

x = np.linspace(-2 * np.pi, 2 * np.pi, N).reshape(-1, 1)
x_train = torch.tensor(x, dtype=torch.float32)

optimizer.zero_grad()

for param in model.parameters():
    print(param)
