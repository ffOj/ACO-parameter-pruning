import torch
from torchvision.datasets import MNIST
from torchvision.transforms import *
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

# transforms = Compose([
#     ToTensor(),
#     Lambda(lambda x: x.flatten() / 255),
# ])

# def poisson_ants(x, size):
#     num_ants = np.zeros(x.shape)
#     for i, x_ in enumerate(x):
#         # print(x_.shape)
#         num_ants[i] = np.sum(np.random.poisson(x_, size))
#     return num_ants

# # train_dataset = MNIST(root='/Users/joshuaoffergeld/Desktop/data', train=True, transform=transforms)
# # test_dataset = MNIST(root='/Users/joshuaoffergeld/Desktop/data', train=False, transform=transforms)

# DATASET_PATH = "./data"
# train_dataset = MNIST(root=DATASET_PATH,
#                       download=True,
#                       train=True,
#                       transform=transforms)
# test_dataset = MNIST(root=DATASET_PATH,
#                      download=True,
#                      train=False,
#                      transform=transforms)

# train_dataloader = torch.utils.data.DataLoader(train_dataset)  # TODO: make batching possible
# test_dataloader = torch.utils.data.DataLoader(test_dataset)

# # print(next(iter(train_dataloader))[0].shape)
# img = torch.flatten(next(iter(train_dataloader))[0])
# input_ants = poisson_ants(img, 1000)
# img_ants = np.reshape(input_ants, (28, 28))

# reshape_img = np.reshape(img, (28, 28))

# fig, ax = plt.subplots(2, 1, figsize = (12, 12))
# # ax[0].imshow(reshape_img)
# # ax[0].set_colorbar()
# # ax[1].imshow(img_ants)
# fig.colorbar(ax[0].imshow(reshape_img), ax = ax[0])
# fig.colorbar(ax[1].imshow(img_ants), ax = ax[1])

# plt.show()

n_inputs, n_outputs = 784, 16

weights = torch.rand((n_inputs, n_outputs))
weights = weights * (1 / weights.sum(dim=1).reshape(-1, 1))

# print(weights)

# n_inputs, n_outputs = 16, 10
# weights = torch.rand((n_inputs, n_outputs))
# weights = weights * (1 / weights.sum(dim=1).reshape(-1, 1))

# print(torch.sum(weights, axis = 0))
print(weights[0])
print(np.random.multinomial(0.0, weights[0, :]))

