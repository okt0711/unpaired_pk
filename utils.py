import torch
from torch.nn import init
from torch.optim import lr_scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'constant':
                init.constant_(m.weight.data, init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def DCE2img(Ct, min=None, max=None):
    Ct1 = Ct[0, 6:7, :, :]
    Ct2 = Ct[0, 7:8, :, :]
    Ct3 = Ct[0, 8:9, :, :]
    Ct4 = Ct[0, 9:10, :, :]
    img = torch.cat([Ct1, Ct2, Ct3, Ct4], 1)
    if min == None:
        min = torch.min(img)
    if max == None:
        max = torch.max(img)
    img = torch.clamp(img, min, max)
    img = (img - min) / (max - min) * 255
    img = img.type(torch.ByteTensor)
    return img


def params2img(params, s1, s2, s3=None):
    if params.size(1) == 2:
        max_Ktrans = 0.005
        max_vp = 0.1
        Ktrans = torch.clamp(params[0, 0:1, :, :] / s1, 0, max_Ktrans)
        vp = torch.clamp(params[0, 1:2, :, :] / s2, 0, max_vp)
        Ktrans = Ktrans / max_Ktrans * 255
        vp = vp / max_vp * 255
        img = torch.cat([Ktrans, vp], 1)
    else:
        max_Ktrans = 0.1
        max_vp = 0.05
        max_ve = 0.4
        Ktrans = torch.clamp(params[0, 0:1, :, :] / s1, 0, max_Ktrans)
        vp = torch.clamp(params[0, 1:2, :, :] / s2, 0, max_vp)
        ve = torch.clamp(params[0, 2:3, :, :] / s3, 0, max_ve)
        Ktrans = Ktrans / max_Ktrans * 255
        vp = vp / max_vp * 255
        ve = ve / max_ve * 255
        img = torch.cat([Ktrans, vp, ve], 1)
    img = img.type(torch.ByteTensor)
    return img


class Mean:
    def __init__(self):
        self.numel = 0
        self.mean = 0
        self.val = 0

    def __call__(self, val):
        self.mean = self.mean * (self.numel / (self.numel + 1)) + val / (self.numel + 1)
        self.numel += 1
        self.val = val

    def step(self):
        return self.val

    def epoch(self):
        return self.mean

    def reset(self):
        self.numel = 0
        self.mean = 0
        self.val = 0
