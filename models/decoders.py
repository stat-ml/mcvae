import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out


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
            nn.Linear(in_features=hidden_dim, out_features=450),
            act_func(),
            nn.Linear(in_features=450, out_features=512),
            act_func(),
            View((32, 4, 4)),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5,
                               stride=2, padding=2),
            act_func(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
            act_func(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, z):
        return self.net(z)


class FC_decoder_cifar(nn.Module):
    def __init__(self, act_func, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
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
            nn.BatchNorm2d(hidden_dim),
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
