import torch
from torch import nn
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose, Lambda, ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam

import numpy as np
from matplotlib import pyplot as plt

import os

from network.models import AntModel
from network.modules import ACOLayer
from network.utils import plot_network


def prune_weights(model, transition_matrices, pruning_condition):
    new_dict = dict()
    n_pruned = []
    for key, transitions in zip(model.state_dict().keys(), transition_matrices):
        value = model.state_dict()[key]
        param = torch.where(pruning_condition(transitions.T), 0,
                            value)  # transpose as weights are flipped for ACO networks
        n_pruned.append(torch.sum(param == 0).item())
        new_dict.update({key: param})
    model.load_state_dict(new_dict)
    return n_pruned


def random_prune(model, n_pruned_arr):
    for param, n_prune in zip(model.parameters(), n_pruned_arr):
        x, y = param.shape
        random_weights = np.random.choice(np.arange(x*y), n_prune, replace = False)
        xs = random_weights%x
        ys = random_weights//x
        for i, j in zip(xs, ys):
            param[i, j] = 0.

        return torch.sum(param == 0)


def evaluate(model, data):
    model.eval()
    acc = 0.
    with torch.no_grad():
        for x, y in data:
            out = model(x)
            acc += torch.sum(out.argmax(dim=1) == y).item() / len(x)
    model.train()
    return acc / len(data)


def pruning_condition(x):
    return x < 1


# init dataset
DATA_PATH = '~/Desktop/data/'  # SET THE PATH TO THE DATA

transform = Compose([
    ToTensor(),
    Lambda(lambda x: x.flatten() / 255)
])
train_dataset = FashionMNIST(root=DATA_PATH, train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root=DATA_PATH, train=False, transform=transform, download=True)

batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# init classifier
classifier = nn.Sequential(
    nn.Linear(784, 16, bias=False),
    nn.ReLU(),
    nn.Linear(16, 10, bias=False)
)

# train classifier
MODEL_PATH = 'model-params/fashion-classifier.pth'
if os.path.exists(MODEL_PATH):
    classifier.load_state_dict(torch.load(MODEL_PATH))
else:
    N_EPOCHS = 5
    optim = Adam(classifier.parameters(), lr=0.05)
    loss_fn = nn.CrossEntropyLoss()
    accs = []
    losss = []
    for epoch in range(N_EPOCHS):
        for x, y in train_dataloader:
            optim.zero_grad()
            out = classifier(x)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()

            losss.append(loss.item())
        acc = evaluate(classifier, test_dataloader)
        accs.append(acc)
        print(f'Accuracy of Epoch {epoch + 1}: {acc}')

    plt.plot(np.arange(1, len(accs)+1) * len(train_dataloader), accs,
             label='test accuracy', linestyle='--', marker='o')
    plt.plot(losss, label='train loss')
    plt.xlabel('iterations')
    plt.title('Performance of ACO weight pruning')
    plt.legend()
    plt.savefig('plots/fashion-classifier-training.png')
    plt.show()

    torch.save(classifier.state_dict(), MODEL_PATH)

# store parameters for later weight pruning
classifier.requires_grad_(False)

state_dict = dict()
for key in classifier.state_dict().keys():
    state_dict.update({key: torch.clone(classifier.state_dict()[key])})

total_num_weights = 0
for weight in classifier.parameters():
    total_num_weights += weight.shape[0] * weight.shape[1]


# init ACO
rho = 1e-1
min_acc = 0.6

n_iter = 50
n_ants = 1000
n_trials = 5


inputs = torch.full(size=[1, 1], fill_value=n_ants)

pruner = AntModel([
    ACOLayer(1, 784, rho),
    ACOLayer(784, 16, rho),
    ACOLayer(16, 10, rho)
])

aco_accs = []
baseline_accs = []
pruned_perc = []
pruned_perc_random = []
for i in range(n_trials):
    for iter in range(n_iter):
        for x, y in test_dataloader:
            pruner(inputs)
            transition_matrices = pruner.get_transition_matrices()
            n_pruned_arr = prune_weights(classifier, transition_matrices[1:],
                                     pruning_condition)  # since ACO connects from 1 -> n_inputs, we remove these
            n_pruned = np.sum(n_pruned_arr)

            out = classifier(x)
            aco_acc = torch.sum(torch.argmax(out, dim=1) == y).item() / len(x)

            if aco_acc > min_acc:
                pruner.update(aco_acc, transition_matrices)

            classifier.load_state_dict(state_dict)  # reset connections / undo pruning

            n_pruned_random = random_prune(classifier, n_pruned_arr)
            out = classifier(x)
            baseline_acc = torch.sum(torch.argmax(out, dim=1) == y).item() / len(x)

            classifier.load_state_dict(state_dict)  # reset connections / undo pruning

            print(f'Iteration: {iter}, ACO Accuracy: {aco_acc}, Baseline Accuracy: {baseline_acc}, Pruned Weights: {n_pruned / total_num_weights * 100}%, Random Pruned Weights: {n_pruned_random / total_num_weights * 100}')

            aco_accs.append(aco_acc)
            baseline_accs.append(baseline_acc)
            pruned_perc.append(np.sum(n_pruned) / total_num_weights)
            pruned_perc_random.append(n_pruned_random/ total_num_weights)

        # plot_network(pruner)

    pruned_perc = np.array(pruned_perc)
    aco_accs = np.array(aco_accs)
    baseline_accs = np.array(baseline_accs)
    np.savetxt(f'./data/fashion-pruned_percentage-accuracies_aco_baseline-{n_ants}-1.0-{rho}-{min_acc}-{i}.txt', (pruned_perc, aco_accs, baseline_accs), delimiter=',')  # for historical reasons, there is a 1.0 in the file path - you may delete it but be aware in the analysis

    plt.figure()
    plt.plot(aco_accs, label='ACO accuracy', alpha=0.5)
    plt.plot(baseline_accs, label='baseline accuracy', alpha=0.5)
    plt.plot(pruned_perc, label='proportion of weights pruned')
    plt.xlabel('iterations')
    plt.title('Performance of ACO weight pruning')
    plt.legend()
    plt.savefig(f'./plots/fashion-pruning-perf-{n_ants}-{rho}-{min_acc}-{i}.png')
    plt.show()
