import torch
import torch.nn as nn


class dncnn(nn.Module):
    def __init__(self, channels=1, num_classes=1, dim=27):
        super(dncnn, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        num_of_layers = 17
        layers = []
        layers.append(nn.Conv1d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv1d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm1d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv1d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

        self.dim = dim

        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        out = self.dncnn(x)

        if self.num_classes > 1:
            out = out.view(-1, self.dim * 1)
            out = self.fc_layers(out)

        return out


if __name__ == '__main__':

    with torch.no_grad():
        
        num_classes = 1
        net = dncnn(num_classes=num_classes, dim=256).cuda()

        x = torch.randn(64, 1, 256).cuda()
        y = net(x)
        print(y.shape)
