import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class conv_norm_act(nn.Module):
    def __init__(self, in_c, out_c, k, pad=0, bias=True, act_name='relu'):
        super(conv_norm_act, self).__init__()

        self.conv = nn.Conv1d(in_c, out_c, kernel_size=k, padding=pad, bias=bias)
        self.norm = nn.BatchNorm1d(out_c)

        if act_name == 'mish':
            self.act = Mish()
        elif act_name == 'lelu':
            self.act = nn.LeakyReLU()
        elif act_name == 'pelu':
            self.act = nn.PReLU()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        return x


class mkm_bone(nn.Module):
    def __init__(self, act_name='relu', bias=False):
        super(mkm_bone, self).__init__()

        self.conv_in = conv_norm_act(1, 64, k=3, pad=1, bias=bias, act_name=act_name)

        self.conv_1_2 = conv_norm_act(64, 64, k=1, pad=0, bias=bias, act_name=act_name)
        self.conv_1_3 = conv_norm_act(64, 64, k=3, pad=1, bias=bias, act_name=act_name)
        self.conv_1_4 = conv_norm_act(64, 64, k=5, pad=2, bias=bias, act_name=act_name)

        self.conv_out = conv_norm_act(64, 1, k=3, pad=1, bias=bias, act_name=act_name)

    def forward(self, x):
        x = self.conv_in(x)

        x11 = self.conv_1_2(x)
        x12 = self.conv_1_3(x)
        x13 = self.conv_1_4(x)
        x1 = x11 + x12 + x13

        x1 = self.conv_out(x1)
        return x1

class mdm_bone(nn.Module):
    def __init__(self, dim=27):
        super(mdm_bone, self).__init__()

        self.down1 = nn.Sequential(nn.Linear(dim, 128), nn.ReLU())
        self.down2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.down3 = nn.Sequential(nn.Linear(64, 32), nn.ReLU())

        self.up1 = nn.Sequential(nn.Linear(32, 64), nn.ReLU())
        self.up2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        self.up3 = nn.Sequential(nn.Linear(128, dim), nn.ReLU())

    def forward(self, x):

        d1 = self.down1(x)
        # print(d1.shape)

        d2 = self.down2(d1)
        # print(d2.shape)

        d3 = self.down3(d2)
        # print(d3.shape)

        u1 = self.up1(d3)
        u1 = u1 + d2
        # print(u1.shape)

        u2 = self.up2(u1)
        u2 = u2 + d1
        # print(u2.shape)

        u3 = self.up3(u2)
        u3 = u3 + x
        # print(u3.shape)

        return u3


class ourcf(nn.Module):
    def __init__(self, in_c=1, act_name='mish', bias=False, num_classes=1, dim=27):
        super(ourcf, self).__init__()

        self.bone1 = mkm_bone(act_name=act_name, bias=bias)
        self.bone2 = mdm_bone(dim=dim)

        self.activete = nn.Tanh()

        self.att_out = conv_norm_act(in_c * 2, in_c, k=3, pad=1, bias=bias, act_name=act_name)

        self.dim = dim

        self.num_classes = num_classes
        self.fc_layers = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes), nn.Softmax(dim=1))

    def forward(self, x):

        x1 = self.bone1(x)
        x2 = self.bone2(x)
        x2 = x1 + x2

        x3 = torch.cat([x, x2], 1)
        x3 = self.activete(x3)
        x3 = self.att_out(x3)
        x3 = x2 * x3
        # print(x3.shape)
        if self.num_classes > 1:
            x3 = x3.view(-1, self.dim * 1)
            x3 = self.fc_layers(x3)

        return x3


if __name__ == '__main__':

    with torch.no_grad():

        num_classes = 4
        net = ourcf(num_classes=num_classes).cuda()
        print(net)
        x = torch.randn(64, 1, 27).cuda()
        y = net(x)
        print('y', y.shape)
