from torchvision import transforms, datasets
import torch


def make_dataloaders(dataset, batch_size, val_batch_size, **kwargs):
    if dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=val_batch_size, shuffle=False, **kwargs)

    elif dataset == 'fashionmnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=True, download=True,
                                  transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=val_batch_size, shuffle=False, **kwargs)
    elif dataset == 'cifar':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=True, download=True,
                             transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data', train=False, transform=transforms.ToTensor()),
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
