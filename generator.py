import torch
import torch.nn as nn
from ops import LocalPathBlock, GlobalPathBlock, StartConvBlock, LastConvBlock, CpBlock
import numpy as np


def inv_sigmoid(x):
    return -torch.log((1 / (x + 1e-8)) - 1)


class Dualnet(nn.Module):
    def __init__(self, in_channels, out_channels, opt):
        super(Dualnet, self).__init__()
        self.start_conv = StartConvBlock(in_channels, opt.ngf, 'none', 'relu')

        self.local_path = LocalPathBlock(opt.ngf, 'none', 'relu')
        self.global_path = GlobalPathBlock(opt.ngf, 'none', 'relu')

        self.last_conv = LastConvBlock(opt.ngf * 2, out_channels, 'none', 'lrelu')
        self.cp_block = CpBlock(opt.ngf * 2, in_channels, 'none', 'lrelu')

    def forward(self, x):
        feature = self.start_conv(x)
        local_feature = self.local_path(feature)
        global_feature = self.global_path(feature)
        cat = torch.cat([local_feature, global_feature], dim=1)
        out = self.last_conv(cat)
        out_cp = self.cp_block(cat)
        return out, out_cp


class patlak_model(nn.Module):
    def __init__(self, opt):
        super(patlak_model, self).__init__()
        self.deltt = opt.deltt
        self.n_time = opt.n_time
        self.TR = opt.TR
        self.FA = opt.FA
        self.r1 = opt.r1
        self.eps = 1e-8
        self.scale_ktrans = opt.scale_ktrans
        self.scale_vp = opt.scale_vp
        self.device = torch.device('cuda:' + str(opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')

    def forward(self, params, Cp, S0, T1):
        Ktrans = params[:, 0:1, :, :] / self.scale_ktrans / 60  # min^(-1) -> sec^(-1)
        vp = params[:, 1:2, :, :] / self.scale_vp
        Cp = Cp.unsqueeze(2).unsqueeze(3).expand([-1, -1, Ktrans.size(2), Ktrans.size(3)])

        Ce = torch.zeros_like(Cp)
        for n in range(self.n_time):
            Ce[:, n, :, :] = torch.sum(Cp[:, :n + 1, :, :], dim=1) * self.deltt
        Ct = vp * Cp + Ktrans * Ce

        P = self.TR / (T1 + self.eps)
        Q = self.r1 * (self.TR / 1000) * Ct
        cos = np.cos(self.FA * np.pi / 180)
        St = ((1 - torch.exp(-P - Q)) * (1 - cos * torch.exp(-P))) / ((1 - cos * torch.exp(-P - Q)) * (1 - torch.exp(-P)) + self.eps) * S0

        return St


class etofts_model(nn.Module):
    def __init__(self, opt):
        super(etofts_model, self).__init__()
        self.deltt = opt.deltt
        self.n_time = opt.n_time
        self.TR = opt.TR
        self.FA = opt.FA
        self.r1 = opt.r1
        self.eps = 1e-8
        self.scale_ktrans = opt.scale_ktrans
        self.scale_vp = opt.scale_vp
        self.scale_ve = opt.scale_ve
        self.device = torch.device('cuda:' + str(opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
        self.time_vec = torch.arange(0, self.n_time * self.deltt, self.deltt).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(self.device)

    def forward(self, params, Cp, S0, T1):
        Ktrans = params[:, 0:1, :, :] / self.scale_ktrans / 60  # min^(-1) -> sec^(-1)
        vp = params[:, 1:2, :, :] / self.scale_vp
        ve = params[:, 2:, :, :] / self.scale_ve
        Cp = Cp.unsqueeze(2).unsqueeze(3).expand([-1, -1, Ktrans.size(2), Ktrans.size(3)])

        Kep = Ktrans / (ve + self.eps)

        Ce = torch.zeros_like(Cp)
        for n in range(self.n_time):
            Ce[:, n, :, :] = torch.sum(Cp[:, :n + 1, :, :] * torch.exp(-Kep[:, :1, :, :] * (n * self.deltt - self.time_vec[:, :n + 1, :, :])), dim=1) * self.deltt
        Ct = vp * Cp + Ktrans * Ce

        P = self.TR / (T1 + self.eps)
        Q = self.r1 * self.TR * Ct / 1000
        cos = np.cos(self.FA * np.pi / 180)
        St = ((1 - torch.exp(-P - Q)) * (1 - cos * torch.exp(-P))) / ((1 - cos * torch.exp(-P - Q)) * (1 - torch.exp(-P)) + self.eps) * S0# * (T1 > 0).float()

        return St
