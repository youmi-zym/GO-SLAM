import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from .modules import ConvGRU, CorrBlock, BasicEncoder, GradientClip


def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2).contiguous()
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, kernel_size=(3, 3), padding=(1, 1))
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2, keepdim=False)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1).contiguous()
    up_data = up_data.reshape(batch, 8*ht, 8*wd, dim)

    return up_data


def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch*num, ht, wd, 1)
    mask = mask.view(batch*num, -1, ht, wd)

    return cvx_upsample(disp, mask).view(batch, num, 8*ht, 8*wd)


class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=(3, 3), padding=(1, 1)),
            GradientClip(),
            nn.Softplus(),
        )

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8*8*9, kernel_size=(1, 1), padding=(0, 0))
        )

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch*num, ch, ht, wd)

        _, ix = torch.unique(ii, sorted=True, return_inverse=True)
        net = self.relu(self.conv1(net))

        net =net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8*8*9, ht, wd)

        return 0.01 * eta, upmask


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2*3+1)**2

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, kernel_size=(1, 1), padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, kernel_size=(7, 7), padding=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=(3, 3), padding=(1, 1)),
            GradientClip(),
            nn.Sigmoid(),
        )

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=(3, 3), padding=(1, 1)),
            GradientClip(),
        )

        self.gru = ConvGRU(128, 128+128+64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """ update operation """

        batch, num, ch, ht, wd = net.shape
        device = net.device

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=device)

        out_dim = (batch, num, -1, ht, wd)

        net = net.view(batch*num, -1, ht, wd)
        inp = inp.view(batch*num, -1, ht, wd)
        corr = corr.view(batch*num, -1, ht, wd)
        flow = flow.view(batch*num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*out_dim)
        weight = self.weight(net).view(*out_dim)

        delta = delta.permute(0, 1, 3, 4, 2)[..., :2].contiguous()
        weight = weight.permute(0, 1, 3, 4, 2)[..., :2].contiguous()

        net = net.view(*out_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(device))
            return net, delta, weight, eta, upmask
        else:
            return net, delta, weight


class DroidNet(nn.Module):
    def __init__(self):
        super(DroidNet, self).__init__()
        self.fnet = BasicEncoder(out_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(out_dim=256, norm_fn='none')
        self.update = UpdateModule()

