import torch
from torch import nn
import numpy as np

class ACOLayer(nn.Module):
    """
    Ant-colony-optimization layer

    Every spike represents one ant, every neuron a node in the environment of the ants.
    By default, no ant dies along the way, i.e. in_flow == out_flow.
    Activations route the ants to other nodes. The weights of this layer therefore
    determine the likelihood of an ant to go to a connected node.

    We are not yet enforcing that two ants cannot be in the same location per timestep!
    """
    def __init__(self, n_inputs, n_outputs, rho, relu_cutoff=0):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.weights = torch.full((n_inputs, n_outputs), fill_value=1/n_outputs, dtype=torch.float32)
        self.weights = nn.parameter.Parameter(self.weights, requires_grad=True)

        self.input = None
        self.trail = None
        self.reset_trails()

        self.rho = rho
        self.relu_cutoff = relu_cutoff

    @staticmethod
    def _route_ants(weights, ants):
        bs, n_in = ants.shape

        res = []
        for batch in range(bs):
            paths = []
            for i, (w, inp) in enumerate(zip(weights, ants.T)):
                if inp[batch] == 0:
                    paths.append([])
                    continue
                draw = torch.multinomial(torch.Tensor(w), inp[batch], True)

                paths.append(draw)
            res.append(paths)
        return res

    def normalize(self):
        return self.weights / self.weights.sum(dim=1)[..., None]

    def forward(self, x):
        self.input = x

        weights = self.normalize()

        res = self._route_ants(
            weights.detach().numpy(),
            x.detach().numpy()
        )

        return res
    
    def relu_func(self, x):
        return torch.where(x < self.relu_cutoff, 0, x)

    def update_trails(self, ants):
        self.trail += ants

    def reset_trails(self):
        self.trail = torch.zeros_like(self.weights)

    def backward(self):
        self.weights = nn.parameter.Parameter(
            (1-self.rho) * self.weights + self.rho * self.trail,
            requires_grad=self.weights.requires_grad
        )

        self.reset_trails()
