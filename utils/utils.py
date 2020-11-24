from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data, transform=None, binarize=False):
        super(MyDataset, self).__init__()
        if not isinstance(data, np.ndarray):
            self.data = data.numpy()
        else:
            self.data = data
        self.transform = transform
        self.binarize = binarize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        if self.transform:
            sample = self.transform(sample)
        if self.binarize:
            sample = torch.distributions.Bernoulli(probs=sample).sample()
        return sample, -1.


def make_dataloaders(dataset, batch_size, val_batch_size, binarize=False, **kwargs):
    if dataset == 'mnist':
        train_loader = DataLoader(
            MyDataset(datasets.MNIST('./data', train=True, download=True).train_data, transform=transforms.ToTensor(),
                      binarize=binarize), batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(
            MyDataset(datasets.MNIST('./data', train=False).test_data, transform=transforms.ToTensor(),
                      binarize=binarize),
            batch_size=val_batch_size, shuffle=False, **kwargs)

    elif dataset == 'fashionmnist':
        train_loader = DataLoader(
            MyDataset(datasets.FashionMNIST('./data', train=True, download=True).train_data,
                      transform=transforms.ToTensor(),
                      binarize=binarize),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(
            MyDataset(datasets.FashionMNIST('./data', train=False).test_data, transform=transforms.ToTensor(),
                      binarize=binarize),
            batch_size=val_batch_size, shuffle=False, **kwargs)
    elif dataset == 'cifar':
        train_loader = DataLoader(
            MyDataset(datasets.CIFAR10('./data', train=True, download=True).data, binarize=binarize,
                      transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(
            MyDataset(datasets.CIFAR10('./data', train=False).data, transform=transforms.ToTensor(), binarize=binarize),
            batch_size=val_batch_size, shuffle=False, **kwargs)
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
    }
