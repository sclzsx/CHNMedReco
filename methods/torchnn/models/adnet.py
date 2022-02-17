import torch
import torch.nn as nn


# 只有最后一层bias=True，其余为False
class adnet(nn.Module):
    def __init__(self, in_c=1, out_c=1, num_classes=1, dim=27):
        super(adnet, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv1d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.conv1_2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                     nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, bias=False, dilation=2),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
                                      nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.conv1_16 = nn.Conv1d(in_channels=64, out_channels=in_c, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv1d(in_channels=in_c * 2, out_channels=in_c, kernel_size=1, stride=1, padding=0, groups=1,
                               bias=True)
        self.Tanh = nn.Tanh()

        self.dim = dim

        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        # print(x.shape)
        x1 = self.conv1_1(x)
        # print(x1.shape)
        x1 = self.conv1_2(x1)
        # print(x1.shape)
        x1 = self.conv1_3(x1)
        # print(x1.shape)
        x1 = self.conv1_4(x1)
        # print(x1.shape)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1 = self.conv1_8(x1)
        x1 = self.conv1_9(x1)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out2 = out * x1
        # out2 = x - out

        if self.num_classes > 1:
            out2 = out2.view(-1, self.dim * 1)
            out2 = self.fc_layers(out2)

        return out2


if __name__ == '__main__':

    with torch.no_grad():
        
        num_classes = 1
        net = adnet(num_classes=num_classes, dim=256).cuda()

        x = torch.randn(64, 1, 256).cuda()
        y = net(x)
        print(y.shape)
