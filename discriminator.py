import torch.nn as nn
from ops import CkBlock


class PatchGAN(nn.Module):
    def __init__(self, in_channels, opt):
        super(PatchGAN, self).__init__()
        self.layers = nn.Sequential(
            CkBlock(in_channels, opt.ndf, 'none', 'lrelu', 2),
            CkBlock(opt.ndf, opt.ndf * 2, 'instance', 'lrelu', 2),
            CkBlock(opt.ndf * 2, opt.ndf * 4, 'instance', 'lrelu', 2),
            CkBlock(opt.ndf * 4, opt.ndf * 8, 'instance', 'lrelu', 1),
            CkBlock(opt.ndf * 8, 1, 'none', 'none', 1)
        )

    def forward(self, x):
        out = self.layers(x)
        return out
