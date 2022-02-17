import torch
from torch import nn


class classifier(nn.Module):
    def __init__(self, num_classes=4, dim=27):
        super(classifier, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv1d(1, dim, 3, stride=1, bias=True),
                                    nn.ReLU(inplace=True), nn.MaxPool1d(9, 3))
        self.layer2 = nn.Sequential(nn.Conv1d(dim, 128, 3, stride=1, bias=True),
                                    nn.ReLU(inplace=True), nn.MaxPool1d(9, 3))
        self.layer3 = nn.Sequential(nn.Conv1d(128, 128, 3, stride=1, bias=True),
                                    nn.ReLU(inplace=True), nn.MaxPool1d(9, 3))
        self.layer4 = nn.Sequential(nn.Conv1d(128, 128, 3, stride=1, bias=True),
                                    nn.ReLU(inplace=True))

        self.dense1 = nn.Sequential(nn.Linear(128 * 3, 128), nn.ReLU(), nn.Dropout(0.5))
        self.dense2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5))
        self.dense3 = nn.Sequential(nn.Linear(64, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        print('before view:', x.shape)
        x = x.view(-1, 128 * 3)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


if __name__ == '__main__':

    with torch.no_grad():

        num_classes = 2
        net = classifier(num_classes=2, dim=27).cuda()

        x = torch.randn(4, 1, 27).cuda()
        y = net(x)
        print(y.shape)