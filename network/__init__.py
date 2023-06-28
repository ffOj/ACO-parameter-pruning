import torch
from torchvision.transforms import *
from torch import nn

from network.utils import run, poisson_ants

from network.modules import ACOLayer
from network.models import AntModel

from network.data import NClassDataset, XORDataset

l1 = ACOLayer(2, 5, rho=.1, relu_cutoff=20)
l2 = ACOLayer(5, 5, rho=.1, relu_cutoff=20)
l3 = ACOLayer(5, 2, rho=.1, relu_cutoff=20)

model = AntModel([l1, l2, l3])
optim = torch.optim.SGD(model.parameters(), lr=10)

transforms = Compose([
    Lambda(lambda x: poisson_ants(torch.Tensor(x), N_ANTS, N_TIMESTEPS))
])

# two-class dataset
means = [[5, 3], [3, 5]]
covs = [[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]]
train_dataset = NClassDataset(100, means, covs, transform=transforms)
test_dataset = NClassDataset(1, means, None, transform=transforms)

# XOR data
# means_c1, covs_c1 = [[5, 3], [5, 8]], [[[0.1,0],[0,0.1]], [[0.1,0],[0,0.1]]]
# means_c2, covs_c2 = [[3, 5], [8, 5]], [[[0.1,0],[0,0.1]], [[0.1,0],[0,0.1]]]

# means_c1, covs_c1 = [[3, 3], [13, 13]], [[[0.1, 0], [0, 0.1]], [[0.1, 0], [0, 0.1]]]
# means_c2, covs_c2 = [[8, 13], [3, 13]], [[[0.1, 0], [0, 0.1]], [[0.1, 0], [0, 0.1]]]
# train_dataset = XORDataset(50, means_c1, means_c2, covs_c1, covs_c2, transform=transforms)
# test_dataset = XORDataset(1, means_c1, means_c2, None, None, transform=transforms)

N_ANTS = 100
BATCH_SIZE = 1
N_TIMESTEPS = 1

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

criter = nn.CrossEntropyLoss()
if __name__ == '__main__':
    run(model, optim, train_dataloader, test_dataloader, criter, n_epochs=10, nt=N_TIMESTEPS)
