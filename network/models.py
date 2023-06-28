import torch
from torch import nn
import numpy as np



class AntModel(nn.Module):
    def __init__(self, layers):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.tables = None

    def forward(self, x):
        x = x.to(dtype=torch.int)
        # sum ants over images of batch
        n_ants = torch.sum(x, dim=1)
        # create a list of tables with one table per image
        table = [np.zeros((int(n_ants[i].item()), len(self.layers) + 1), dtype=int) for i in range(len(x))]

        # input layer distribution
        for n_batch, batch in enumerate(x):
            distribution = [i for i, n_ants_per_node in enumerate(batch) for _ in range(n_ants_per_node)]
            table[n_batch][:, 0] = distribution

        # subsequent layers distributions
        for i, layer in enumerate(self.layers, start=1):
            paths = layer(x)

            for n_batch, batch in enumerate(paths):
                for n_neuron, neuron in enumerate(batch):
                    if len(neuron) == 0: continue
                    # find and enter destinations from ants at neurons 0-N sampled by each layer
                    table[n_batch][table[n_batch][:, i-1] == n_neuron, i] = neuron

            x = torch.zeros((len(table), layer.n_outputs), dtype=torch.int)
            for j, sub_table in enumerate(table):
                for idx in sub_table[:, i]:
                    x[j, idx] += 1

            x = layer.relu_func(x)
            for j, ant_vec in enumerate(x):
                for zeroed in np.where(ant_vec == 0)[0]:
                    table[j] = table[j][table[j][:, i] != zeroed]

        self.tables = table
        return x.to(dtype=torch.float32)

    def get_transition_matrices(self):
        transition_matrices = []
        for table in self.tables:
            for i, layer in reversed(list(enumerate(self.layers))):
                transitions = table[:, i:i + 2]
                transitions_matrix = torch.zeros_like(layer.weights)
                for i, j in transitions:
                    transitions_matrix[i, j] += 1

                transition_matrices.append(transitions_matrix)
        return list(reversed(transition_matrices))

    def update(self, pheromone_value, transition_matrices):
        for layer, transitions in zip(self.layers, transition_matrices):
            layer.update_trails(transitions * pheromone_value)
            layer.backward()

    def backward(self, labels):
        connection_matrices = []
        for table, label in zip(self.tables, labels):
            table = table[table[:, -1] == label.item()]
            for i, layer in reversed(list(enumerate(self.layers))):
                connections = table[:, i:i+2]
                connection_matrix = torch.zeros_like(layer.weights)
                for i, j in connections:
                    connection_matrix[i, j] += 1

                connection_matrices.append(connection_matrix)

                layer.update_trails(connection_matrix)

        for layer in self.layers:
            layer.backward()
