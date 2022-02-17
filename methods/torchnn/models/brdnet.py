import torch
import torch.nn as nn


# 全部bias=True
class brdnet(nn.Module):
    def __init__(self, in_c=1, num_classes=1, dim=27):
        super(brdnet, self).__init__()
        self.conv_1_1 = nn.Sequential(nn.Conv1d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.conv_1_2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_4 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_5 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_6 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_7 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_8 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.conv_1_9 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_10 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_11 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_12 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_13 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_14 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_15 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))
        self.conv_1_16 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.conv_1_17 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=in_c, kernel_size=3, padding=1))
        ###############################
        self.conv_2_1 = nn.Sequential(nn.Conv1d(in_channels=in_c, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.conv_2_2 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_3 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_4 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_5 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_6 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_7 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_8 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))

        self.conv_2_9 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.conv_2_10 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_11 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_12 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_13 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_14 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))
        self.conv_2_15 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=2), nn.ReLU(inplace=True))

        self.conv_2_16 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(inplace=True))

        self.conv_2_17 = nn.Sequential(nn.Conv1d(in_channels=64, out_channels=in_c, kernel_size=3, padding=1))
        ###############################
        self.conv_6_to_3 = nn.Sequential(nn.Conv1d(in_channels=in_c * 2, out_channels=in_c, kernel_size=3, padding=1))

        self.dim = dim

        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes), nn.Softmax(dim=1))

    def forward(self, x):
        x1 = self.conv_1_1(x)
        x1 = self.conv_1_2(x1)
        x1 = self.conv_1_3(x1)
        x1 = self.conv_1_4(x1)
        x1 = self.conv_1_5(x1)
        x1 = self.conv_1_6(x1)
        x1 = self.conv_1_7(x1)
        x1 = self.conv_1_8(x1)
        x1 = self.conv_1_9(x1)
        x1 = self.conv_1_10(x1)
        x1 = self.conv_1_11(x1)
        x1 = self.conv_1_12(x1)
        x1 = self.conv_1_13(x1)
        x1 = self.conv_1_14(x1)
        x1 = self.conv_1_15(x1)
        x1 = self.conv_1_16(x1)
        x1 = self.conv_1_17(x1)
        # x1 = x - x1

        y1 = self.conv_2_1(x)
        y1 = self.conv_2_2(y1)
        y1 = self.conv_2_3(y1)
        y1 = self.conv_2_4(y1)
        y1 = self.conv_2_5(y1)
        y1 = self.conv_2_6(y1)
        y1 = self.conv_2_7(y1)
        y1 = self.conv_2_8(y1)
        y1 = self.conv_2_9(y1)
        y1 = self.conv_2_10(y1)
        y1 = self.conv_2_11(y1)
        y1 = self.conv_2_12(y1)
        y1 = self.conv_2_13(y1)
        y1 = self.conv_2_14(y1)
        y1 = self.conv_2_15(y1)
        y1 = self.conv_2_16(y1)
        y1 = self.conv_2_17(y1)
        # y1 = x - y1

        z = torch.cat([x1, y1], 1)
        z = self.conv_6_to_3(z)
        # z = x - z

        if self.num_classes > 1:
            z = z.view(-1, self.dim * 1)
            z = self.fc_layers(z)

        return z




if __name__ == '__main__':

    with torch.no_grad():
        
        num_classes = 1
        net = brdnet(num_classes=2, dim=27).cuda()

        x = torch.randn(64, 1, 27).cuda()
        y = net(x)
        print(y.shape)