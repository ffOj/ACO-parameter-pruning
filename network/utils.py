from network.models import AntModel

import torch
import matplotlib.pyplot as plt
import numpy as np


def validate_model(model, dataloader, loss_fn, nt):
    acc = 0
    loss = 0
    for x, y in dataloader:
        y_oh = torch.nn.functional.one_hot(y, 2)

        out = torch.zeros((len(x), model.layers[-1].n_outputs))
        for t in range(nt):
            out += model(x[:, :, t])

        acc += torch.sum(out.argmax(dim=1) == y).item()
        loss += loss_fn(out, y_oh.to(dtype=torch.float32)).item()
    return acc, loss

# def plot_network(outputs, x):
#     output = outputs.astype(float)
#     for i, out in enumerate(output.T):
#         output[:, i] = output[:, i]/np.max(out)
#
#     plt.title(x)
#     plt.plot(output.T, color='blue', alpha=0.01)
#     plt.show()


def plot_network(model):
    for i, layer in enumerate(model.layers):
        rs, cs = layer.weights.shape
        weights = layer.normalize()
        for r in range(rs):
            for c in range(cs):
                plt.plot([i, i+1], [r/rs, c/cs], color='blue', alpha=weights[r, c].item())

    plt.show()


def run(model: AntModel, optim, train_dataloader, test_dataloader, loss_fn, nt=1, n_epochs=10):
    for epoch in range(1, n_epochs + 1):
        for batch_nr, (x, y_labels) in enumerate(train_dataloader):
            for t in range(nt):
                out = model(x[:, :, t])
                model.backward(y_labels)
                # optim.step()

        acc, loss = validate_model(model, test_dataloader, loss_fn, nt)
        print(acc, loss)
    for x, y in test_dataloader:
        plot_network(model.tables[0], x[:, :, 0])


def poisson_ants(x, size, nt):
    x = torch.nn.functional.normalize(x, p=1, dim=0) * size / nt
    distribution = torch.zeros((len(x), nt))
    for i in range(nt):
        distribution[:, i] = torch.poisson(x)
    return distribution
