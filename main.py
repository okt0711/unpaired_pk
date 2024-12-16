import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from os import makedirs
from os.path import join, isdir
from dataloader import DCEDataset
from options import Options
from model import DCE_cycle, DCE_supervised, DCE_supervised_physics

opt = Options().parse()
sdir = join(opt.save_path, opt.experiment_name)
opt.log_dir = join(sdir, 'log_dir')
opt.ckpt_dir = join(sdir, 'ckpt_dir')
opt.device = torch.device('cuda:' + str(opt.gpu_ids[0]) if torch.cuda.is_available() else 'cpu')
if not isdir(opt.ckpt_dir):
    makedirs(opt.ckpt_dir)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

dataset_train = DCEDataset(opt, 'train')
dataset_test = DCEDataset(opt, 'test')

dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=2, pin_memory=True)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
dataloader_test.flist = dataset_test.flist_D

if opt.model == 'DCE_cycle':
    model = DCE_cycle(opt)
elif opt.model == 'DCE_supervised':
    model = DCE_supervised(opt)
elif opt.model == 'DCE_supervised_physics':
    model = DCE_supervised_physics(opt)
else:
    raise NotImplementedError('Model [%s] is not implemented' % opt.model)

if opt.training:
    model.train(dataloader_train)

model.test(dataloader_test)
model.test_simul(dataloader_test)
