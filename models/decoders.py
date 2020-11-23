import torch
import torch.nn as nn


class FC_decoder_mnist(nn.Module):
    def __init__(self, act_func, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 400),
            act_func(),
            nn.Linear(400, 784)
        )
        self.hidden_dim = hidden_dim

    def forward(self, z):
        return self.net(z)


class CONV_decoder_mnist(nn.Module):
    def __init__(self, act_func, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=25, kernel_size=5, stride=2),
            act_func(),
            nn.ConvTranspose2d(in_channels=25, out_channels=10, kernel_size=5, stride=2),
            act_func(),
            nn.ConvTranspose2d(in_channels=10, out_channels=1, kernel_size=3, stride=2,
                               output_padding=1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, z):
        z = z.view(-1, self.hidden_dim, 1, 1)
        return self.net(z)


class FC_decoder_cifar(nn.Module):
    def __init__(self, act_func, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 400),
            act_func(),
            nn.Linear(400, 3072)
        )
        self.hidden_dim = hidden_dim

    def forward(self, z):
        return self.net(z)


class CONV_decoder_cifar(nn.Module):
    def __init__(self, act_func, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_dim, out_channels=25, kernel_size=5, stride=2),
            act_func(),
            nn.BatchNorm2d(25),
            nn.ConvTranspose2d(in_channels=25, out_channels=10, kernel_size=5, stride=2),
            act_func(),
            nn.BatchNorm2d(10),
            nn.ConvTranspose2d(in_channels=10, out_channels=5, kernel_size=5, stride=2),
            act_func(),
            nn.BatchNorm2d(5),
            nn.ConvTranspose2d(in_channels=5, out_channels=3, kernel_size=4, stride=1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, z):
        z = z.view(-1, self.hidden_dim, 1, 1)
        return self.net(z)


def get_decoder(net_type, act_func, hidden_dim, dataset):
    if str(dataset).lower().find('mnist') > -1:
        if net_type == 'fc':
            return FC_decoder_mnist(act_func=act_func,
                                    hidden_dim=hidden_dim)
        elif net_type == 'conv':
            return CONV_decoder_mnist(act_func=act_func,
                                      hidden_dim=hidden_dim)
    elif str(dataset).lower().find('cifar') > -1:
        if net_type == 'fc':
            return FC_decoder_cifar(act_func=act_func,
                                    hidden_dim=hidden_dim)
        elif net_type == 'conv':
            return CONV_decoder_cifar(act_func=act_func,
                                      hidden_dim=hidden_dim)
