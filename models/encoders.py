import torch
import torch.nn as nn


class FC_encoder_mnist(nn.Module):
    def __init__(self, act_func, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 400),
            act_func(),
            nn.Linear(400, 2 * hidden_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x)


class CONV_encoder_mnist(nn.Module):
    def __init__(self, act_func, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                      stride=2, padding=2),
            act_func(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,
                      stride=2, padding=2),
            act_func(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5,
                      stride=2, padding=2),
            act_func(),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=450),
            act_func(),
            nn.Linear(in_features=450, out_features=2 * hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class FC_encoder_cifar(nn.Module):
    def __init__(self, act_func, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3072, 400),
            act_func(),
            nn.Linear(400, 2 * hidden_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.net(x)


class CONV_encoder_cifar(nn.Module):
    def __init__(self, act_func, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=5, kernel_size=5),
            act_func(),
            nn.BatchNorm2d(5),
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5),
            act_func(),
            nn.BatchNorm2d(10),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5),
            act_func(),
            nn.BatchNorm2d(20),
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5),
            act_func(),
            nn.Flatten(),
            nn.Linear(5120, 2 * hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


def get_encoder(net_type, act_func, hidden_dim, dataset):
    if str(dataset).lower().find('mnist') > -1:
        if net_type == 'fc':
            return FC_encoder_mnist(act_func=act_func,
                                    hidden_dim=hidden_dim)
        elif net_type == 'conv':
            return CONV_encoder_mnist(act_func=act_func,
                                      hidden_dim=hidden_dim)
    elif str(dataset).lower().find('cifar') > -1:
        if net_type == 'fc':
            return FC_encoder_cifar(act_func=act_func,
                                    hidden_dim=hidden_dim)
        elif net_type == 'conv':
            return CONV_encoder_cifar(act_func=act_func,
                                      hidden_dim=hidden_dim)
