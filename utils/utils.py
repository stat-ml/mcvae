from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data, binarize=False):
        super(MyDataset, self).__init__()
        self.data = data
        self.binarize = binarize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        if self.binarize:
            sample = torch.distributions.Bernoulli(probs=sample).sample()
        return sample, -1.


def make_dataloaders(dataset, batch_size, val_batch_size, binarize=False, net_type='fc', **kwargs):
    if dataset == 'mnist':
        train_data = datasets.MNIST('./data', train=True, download=True).train_data.to(torch.float32)
        train_data /= train_data.max()
        if net_type == 'conv':
            train_data = train_data.unsqueeze(1)
        train_dataset = MyDataset(train_data, binarize=binarize)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

        val_data = datasets.MNIST('./data', train=False).test_data.to(torch.float32)
        val_data /= val_data.max()
        if net_type == 'conv':
            val_data = val_data.unsqueeze(1)
        val_dataset = MyDataset(val_data, binarize=binarize)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, **kwargs)

    elif dataset == 'fashionmnist':
        train_data = datasets.FashionMNIST('./data', train=True, download=True).train_data.to(torch.float32)
        train_data /= train_data.max()
        if net_type == 'conv':
            train_data = train_data.unsqueeze(1)
        train_dataset = MyDataset(train_data, binarize=binarize)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

        val_data = datasets.FashionMNIST('./data', train=False).test_data.to(torch.float32)
        val_data /= val_data.max()
        if net_type == 'conv':
            val_data = val_data.unsqueeze(1)
        val_dataset = MyDataset(val_data, binarize=binarize)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, **kwargs)

    elif dataset == 'cifar':
        train_data = datasets.CIFAR10('./data', train=True, download=True).data.to(torch.float32)
        train_data /= train_data.max()
        train_dataset = MyDataset(train_data, binarize=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

        val_data = datasets.CIFAR10('./data', train=False).data.to(torch.float32)
        val_data /= val_data.max()
        val_dataset = MyDataset(val_data, binarize=False)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, **kwargs)

    elif dataset == 'celeba':
        train_loader = DataLoader(
            datasets.CelebA('./data', split="all", download=True,
                            transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(
            datasets.CelebA('./data', split="valid", transform=transforms.ToTensor()),
            batch_size=val_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError

    return train_loader, val_loader


def get_activations():
    return {
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "tanh": torch.nn.Tanh,
        "logsoftmax": lambda: torch.nn.LogSoftmax(dim=-1),
        "logsigmoid": torch.nn.LogSigmoid,
        "softplus": torch.nn.Softplus,
    }
