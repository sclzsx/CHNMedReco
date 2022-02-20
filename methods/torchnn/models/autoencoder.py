import torch
from torch import nn


class autoencoder(nn.Module):
    def __init__(self, num_classes=1, dim=27):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, dim), nn.Sigmoid())

        self.dim = dim

        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(dim, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        # print(x.shape)

        x = self.encoder(x)
        # print(x.shape)

        x = self.decoder(x)
        # print(x.shape)

        if self.num_classes > 1:
            x = x.view(-1, self.dim * 1)
            x = self.fc_layers(x)

        return x


import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class autoencoder2(nn.Module):
    def __init__(self, num_classes=1, dim=27):
        super(autoencoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 128), Mish(),
            nn.Linear(128, 64), Mish(),
            nn.Linear(64, 32), Mish(),
            nn.Linear(32, 16), Mish(),
            nn.Linear(16, 8), Mish(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), Mish(),
            nn.Linear(16, 32), Mish(),
            nn.Linear(32, 64), Mish(),
            nn.Linear(64, 128), Mish(),
            nn.Linear(128, dim), nn.Sigmoid())

        self.dim = dim

        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(dim, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        # print(x.shape)

        x = self.encoder(x)
        # print(x.shape)

        x = self.decoder(x)
        # print(x.shape)

        if self.num_classes > 1:
            x = x.view(-1, self.dim * 1)
            x = self.fc_layers(x)

        return x


class autoencoder1(nn.Module):
    def __init__(self, num_classes=1, dim=27):
        super(autoencoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 32), Mish(),
            nn.Linear(32, 16), Mish(),
            nn.Linear(16, 8), Mish())
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), Mish(),
            nn.Linear(16, 32), Mish(),
            nn.Linear(32, dim), nn.Sigmoid())

        self.dim = dim

        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(dim, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        # print(x.shape)

        x = self.encoder(x)
        # print(x.shape)

        x = self.decoder(x)
        # print(x.shape)

        if self.num_classes > 1:
            x = x.view(-1, self.dim * 1)
            x = self.fc_layers(x)

        return x


class autoencoder3(nn.Module):
    def __init__(self, num_classes=1, dim=27):
        super(autoencoder3, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, 32), Mish())
        self.linear2 = nn.Sequential(nn.Linear(32, 16), Mish())
        self.linear3 = nn.Sequential(nn.Linear(16, 8), Mish())

        self.linear4 = nn.Sequential(nn.Linear(8, 16), Mish())
        self.linear5 = nn.Sequential(nn.Linear(16, 32), Mish())
        self.linear6 = nn.Sequential(nn.Linear(32, dim), nn.Sigmoid())

        self.dim = dim

        self.num_classes = num_classes
        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(dim, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x1)
        x3 = self.linear3(x2)

        x4 = self.linear4(x3)
        x4 = x4 + x2

        x5 = self.linear5(x4)
        x5 = x5 + x1

        x6 = self.linear6(x5)

        if self.num_classes > 1:
            x6 = x6.view(-1, self.dim * 1)
            x6 = self.fc_layers(x6)

        return x6


class autoencoder4(nn.Module):
    def __init__(self, num_classes=1, dim=27):
        super(autoencoder4, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, 128), Mish())
        self.linear2 = nn.Sequential(nn.Linear(128, 64), Mish())
        self.linear3 = nn.Sequential(nn.Linear(64, 32), Mish())
        self.linear4 = nn.Sequential(nn.Linear(32, 16), Mish())
        self.linear5 = nn.Sequential(nn.Linear(16, 8), Mish())

        self.linear6 = nn.Sequential(nn.Linear(8, 16), Mish())
        self.linear7 = nn.Sequential(nn.Linear(16, 32), Mish())
        self.linear8 = nn.Sequential(nn.Linear(32, 64), Mish())
        self.linear9 = nn.Sequential(nn.Linear(64, 128), Mish())
        self.linear10 = nn.Sequential(nn.Linear(128, dim), nn.Sigmoid())

        self.dim = dim

        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(dim, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        x1 = self.linear1(x)  # 128
        x2 = self.linear2(x1)  # 64
        x3 = self.linear3(x2)  # 32
        x4 = self.linear4(x3)  # 16
        x5 = self.linear5(x4)  # 8

        x6 = self.linear6(x5) + x4  # 16
        x7 = self.linear7(x6) + x3
        x8 = self.linear8(x7) + x2
        x9 = self.linear9(x8)
        x10 = self.linear10(x9)

        if self.num_classes > 1:
            x10 = x.view(-1, self.dim * 1)
            x10 = self.fc_layers(x10)

        return x10


if __name__ == '__main__':
    with torch.no_grad():
        net = autoencoder4(num_classes=2, dim=27).cuda()

        x = torch.randn(64, 1, 27).cuda()
        y = net(x)
        print(y.shape)
