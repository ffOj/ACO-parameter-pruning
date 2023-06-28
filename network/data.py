import torch
from torch.utils.data import Dataset
import numpy as np


class NClassDataset(Dataset):
    def __init__(self, n_datapoints_per_class, mus, covs, transform=None):
        self.samples = []
        self.n_datapoints_per_class = n_datapoints_per_class
        if covs is not None:
            for mu, cov in zip(mus, covs):
                self.samples.extend(list(np.random.multivariate_normal(mean=mu, cov=cov, size=n_datapoints_per_class)))
        else:
            self.samples.extend(mus)

        self.tf = transform

    def __getitem__(self, item):
        out = self.tf(self.samples[item]) if self.tf is not None else self.samples[item]
        return torch.Tensor(out), int(item // self.n_datapoints_per_class)

    def __len__(self):
        return len(self.samples)


class XORDataset(Dataset):
    def __init__(self, n_datapoints_per_class, mus_c1, mus_c2, covs_c1, covs_c2, transform=None):
        self.samples = []
        self.n_datapoints_per_class = n_datapoints_per_class
        self.tf = transform

        if covs_c1 is not None:

            for mu, cov in zip(mus_c1, covs_c1):
                self.samples.extend(list(np.random.multivariate_normal(mean=mu, cov=cov, size=n_datapoints_per_class)))

            for mu, cov in zip(mus_c2, covs_c2):
                self.samples.extend(list(np.random.multivariate_normal(mean=mu, cov=cov, size=n_datapoints_per_class)))
        else:
            self.samples.extend(mus_c1)
            self.samples.extend(mus_c2)

    def __getitem__(self, item):
        out = self.tf(self.samples[item]) if self.tf is not None else self.samples[item]
        return torch.Tensor(out), int(item // (self.n_datapoints_per_class * 2))

    def __len__(self):
        return len(self.samples)
