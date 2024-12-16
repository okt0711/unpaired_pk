import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from os import makedirs
from os.path import join, isdir, isfile
from scipy import io as sio
from math import ceil
from generator import Dualnet, patlak_model, etofts_model
from discriminator import PatchGAN
from utils import init_net, get_scheduler, Mean, set_requires_grad, params2img, DCE2img
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DCE_cycle:
    def __init__(self, opt):
        self.device = opt.device
        self.disp_step = opt.disp_step
        self.lambda_adv = opt.lambda_adv
        self.lambda_cycle = opt.lambda_cycle
        self.lambda_cp = opt.lambda_cp
        self.lambda_tv = opt.lambda_tv
        self.lambda_l1 = opt.lambda_l1
        self.n_time = opt.n_time
        self.kinetic_model = opt.kinetic_model
        if self.kinetic_model == 'patlak':
            self.n_param = 2
        else:
            self.n_param = 3
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.n_epochs = opt.n_epochs
        self.n_epochs_decay = opt.n_epochs_decay
        self.continue_epoch = opt.continue_epoch
        self.test_epoch = opt.test_epoch
        self.lr = opt.lr
        self.init_type = opt.init_type
        self.init_gain = opt.init_gain
        self.save_epoch = opt.save_epoch
        self.save_path = opt.save_path
        self.experiment_name = opt.experiment_name
        self.scale_ktrans = opt.scale_ktrans
        self.scale_vp = opt.scale_vp
        self.scale_ve = opt.scale_ve
        self.opt = opt
        self.ckpt_dir = opt.ckpt_dir
        self.log_dir = opt.log_dir
        self.writer = SummaryWriter(self.log_dir)

        self.adv_loss = nn.MSELoss()
        self.cycle_loss = nn.L1Loss()
        self.loss_name = ['G/total_loss',
                          'G/adv_loss',
                          'G/cycle_DCE_loss',
                          'G/cycle_params_loss',
                          'G/cycle_Cp_loss',
                          'D/total_loss',
                          'D/adv_loss']

        self.G_D2P = Dualnet(self.n_time, self.n_param, self.opt).to(self.device)
        if self.kinetic_model == 'patlak':
            self.G_P2D = patlak_model(self.opt).to(self.device)
        elif self.kinetic_model == 'etofts':
            self.G_P2D = etofts_model(self.opt).to(self.device)
        else:
            raise NotImplementedError('Model [%s] is not implemented' % self.kinetic_model)
        self.D_P = PatchGAN(self.n_param, self.opt).to(self.device)

        self.G_D2P_optim = torch.optim.Adam(self.G_D2P.parameters(), self.lr, betas=(self.beta1, self.beta2))
        self.D_optim = torch.optim.Adam(self.D_P.parameters(), self.lr, betas=(self.beta1, self.beta2))
        self.G_D2P_scheduler = get_scheduler(self.G_D2P_optim, self.opt)
        self.D_scheduler = get_scheduler(self.D_optim, self.opt)

    def train(self, dataloader_train):
        if isfile(join(self.ckpt_dir, str(self.continue_epoch) + '.pth')):
            checkpoint = torch.load(join(self.ckpt_dir, str(self.continue_epoch) + '.pth'))
            self.G_D2P.load_state_dict(checkpoint['G_D2P_state_dict'])
            self.D_P.load_state_dict(checkpoint['D_P_state_dict'])
            self.G_D2P_optim.load_state_dict(checkpoint['G_D2P_optim_state_dict'])
            self.D_optim.load_state_dict(checkpoint['D_optim_state_dict'])
            self.G_D2P_scheduler.load_state_dict(checkpoint['G_D2P_scheduler_state_dict'])
            self.D_scheduler.load_state_dict(checkpoint['D_scheduler_state_dict'])
            trained_epoch = checkpoint['epoch']
            print('Start from saved model - ' + str(trained_epoch))
        else:
            init_net(self.G_D2P, self.init_type, self.init_gain)
            init_net(self.D_P, self.init_type, self.init_gain)
            trained_epoch = 0
            print('Start initially')

        losses = {name: Mean() for name in self.loss_name}
        dataset_train_len = len(dataloader_train)

        for epoch in tqdm(range(trained_epoch, self.n_epochs + self.n_epochs_decay), desc='Epoch', total=self.n_epochs + self.n_epochs_decay, initial=trained_epoch):
            for name in self.loss_name:
                losses[name].reset()
            disp_cnt = 0

            self.G_D2P.train()
            for step, (real_DCE, _, S0_D, T1_D, mask_D, real_params, Cp_P, S0_P, T1_P, mask_P) in enumerate(tqdm(dataloader_train)):
                real_DCE = real_DCE.to(self.device)
                S0_D = S0_D.to(self.device)
                T1_D = T1_D.to(self.device)
                mask_D = mask_D.to(self.device)
                real_params = real_params.to(self.device)
                Cp_P = Cp_P.to(self.device)
                S0_P = S0_P.to(self.device)
                T1_P = T1_P.to(self.device)
                mask_P = mask_P.to(self.device)

                set_requires_grad(self.D_P, False)

                fake_params, Cp_D = self.G_D2P(real_DCE)
                fake_DCE = self.G_P2D(real_params, Cp_P, S0_P, T1_P)

                cycle_params, cycle_Cp_P = self.G_D2P(fake_DCE)
                cycle_DCE = self.G_P2D(fake_params, Cp_D, S0_D, T1_D)

                real_DCE = real_DCE * mask_D
                fake_params = fake_params * mask_D
                cycle_DCE = cycle_DCE * mask_D
                real_params = real_params * mask_P
                fake_DCE = fake_DCE * mask_P
                cycle_params = cycle_params * mask_P

                fake_adv = self.D_P(fake_params)

                G_adv_loss = self.adv_loss(fake_adv, torch.ones_like(fake_adv))
                G_cycle_DCE_loss = self.cycle_loss(cycle_DCE, real_DCE)
                G_cycle_params_loss = self.cycle_loss(cycle_params, real_params)
                G_cycle_Cp_loss = self.cycle_loss(cycle_Cp_P, Cp_P)
                G_total_loss = self.lambda_adv * G_adv_loss + self.lambda_cycle * (G_cycle_DCE_loss + G_cycle_params_loss + self.lambda_cp * G_cycle_Cp_loss)# + self.lambda_l1 * G_l1_norm + self.lambda_tv * G_tv_loss

                self.G_D2P_optim.zero_grad()
                G_total_loss.backward()
                self.G_D2P_optim.step()

                set_requires_grad(self.D_P, True)

                real_adv = self.D_P(real_params)
                fake_adv = self.D_P(fake_params.detach())

                D_adv_loss = self.adv_loss(real_adv, torch.ones_like(real_adv)) + self.adv_loss(fake_adv, torch.zeros_like(fake_adv))
                D_total_loss = D_adv_loss / 2

                self.D_optim.zero_grad()
                D_total_loss.backward()
                self.D_optim.step()

                losses['G/adv_loss'](G_adv_loss.detach())
                losses['G/cycle_DCE_loss'](G_cycle_DCE_loss.detach())
                losses['G/cycle_params_loss'](G_cycle_params_loss.detach())
                losses['G/cycle_Cp_loss'](G_cycle_Cp_loss.detach())
                losses['G/total_loss'](G_total_loss.detach())
                losses['D/adv_loss'](D_adv_loss.detach())
                losses['D/total_loss'](D_total_loss.detach())

                if step % self.disp_step == 0:
                    for name in self.loss_name:
                        self.writer.add_scalar('Step_' + name, losses[name].step(), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('DPD/1_real_D', DCE2img(real_DCE, 0, 1), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('PDP/1_real_P', params2img(real_params, self.scale_ktrans, self.scale_vp, self.scale_ve), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('DPD/2_fake_P', params2img(fake_params, self.scale_ktrans, self.scale_vp, self.scale_ve), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('PDP/2_fake_D', DCE2img(fake_DCE, 0, 1), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('DPD/3_cycle_D', DCE2img(cycle_DCE, 0, 1), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('PDP/3_cycle_P', params2img(cycle_params, self.scale_ktrans, self.scale_vp, self.scale_ve), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    disp_cnt += 1

            for name in self.loss_name:
                self.writer.add_scalar('Epoch_' + name, losses[name].epoch(), epoch + 1)
            self.writer.add_scalar('Epoch/lr', self.G_D2P_scheduler.get_last_lr()[0], epoch + 1)

            self.G_D2P_scheduler.step()
            self.D_scheduler.step()
            if (epoch + 1) % self.save_epoch == 0:
                torch.save({'epoch': epoch + 1, 'G_D2P_state_dict': self.G_D2P.state_dict(),
                            'D_P_state_dict': self.D_P.state_dict(),
                            'G_D2P_optim_state_dict': self.G_D2P_optim.state_dict(),
                            'D_optim_state_dict': self.D_optim.state_dict(),
                            'G_D2P_scheduler_state_dict': self.G_D2P_scheduler.state_dict(),
                            'D_scheduler_state_dict': self.D_scheduler.state_dict()},
                           join(self.ckpt_dir, '{}'.format(epoch + 1) + '.pth'))

    @torch.no_grad()
    def test(self, dataloader):
        checkpoint = torch.load(join(self.ckpt_dir, str(self.test_epoch) + '.pth'))
        self.G_D2P.load_state_dict(checkpoint['G_D2P_state_dict'])
        self.G_D2P.eval()

        save_path = join(self.save_path, self.experiment_name, 'test_unmask_{}'.format(self.test_epoch))
        if not isdir(save_path):
            makedirs(save_path)

        print('Start test!')

        for step, (real_DCE, _, _, _, mask_D, _, _, _, _, _) in enumerate(tqdm(dataloader)):
            real_DCE = real_DCE.to(self.device)
            mask_D = mask_D.to('cpu:0').detach().numpy()

            fake_params, Cp = self.G_D2P(F.pad(real_DCE, [16, 16, 16, 16], "constant", 0))
            fake_params = np.squeeze(fake_params[:, :, 16:-16, 16:-16].to('cpu:0').detach().numpy())
            Cp = np.squeeze(Cp.to('cpu:0').detach().numpy())

            if self.kinetic_model == 'patlak':
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                test_output = {'Ktrans': Ktrans, 'vp': vp, 'mask': mask_D, 'Cp': Cp}
            else:
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                ve = fake_params[2, :, :] / self.scale_ve
                test_output = {'Ktrans': Ktrans, 'vp': vp, 've': ve, 'mask': mask_D, 'Cp': Cp}

            subpath = dataloader.flist[step].split('test/')[1]
            subname = subpath.split('/')[0]
            fname = subpath.split('/')[1]

            sub_save_path = join(save_path, subname)
            if not isdir(sub_save_path):
                makedirs(sub_save_path)

            sio.savemat(join(sub_save_path, fname), test_output)

    @torch.no_grad()
    def test_simul(self, dataloader):
        checkpoint = torch.load(join(self.ckpt_dir, str(self.test_epoch) + '.pth'))
        self.G_D2P.load_state_dict(checkpoint['G_D2P_state_dict'])
        self.G_D2P.eval()

        save_path = join(self.save_path, self.experiment_name, 'test_simul_noise_{}'.format(self.test_epoch))
        if not isdir(save_path):
            makedirs(save_path)

        print('Start test!')

        for step, (_, _, _, _, _, real_params, Cp_P, S0_P, T1_P, mask_P) in enumerate(tqdm(dataloader)):
            real_params = real_params.to(self.device)
            Cp_P = Cp_P.to(self.device)
            S0_P = S0_P.to(self.device)
            T1_P = T1_P.to(self.device)
            mask_P = mask_P.to('cpu:0').detach().numpy()
            simul_DCE = self.G_P2D(real_params, Cp_P, S0_P, T1_P)
            simul_DCE += 0.02 * torch.randn(simul_DCE.size()).to(self.device)

            fake_params, _ = self.G_D2P(F.pad(simul_DCE, [16, 16, 16, 16], "constant", 0))
            fake_params = np.squeeze(fake_params[:, :, 16:-16, 16:-16].to('cpu:0').detach().numpy())

            if self.kinetic_model == 'patlak':
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                test_output = {'Ktrans': Ktrans, 'vp': vp, 'mask': mask_P}
            else:
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                ve = fake_params[2, :, :] / self.scale_ve
                test_output = {'Ktrans': Ktrans, 'vp': vp, 've': ve, 'mask': mask_P}

            subpath = dataloader.flist[step].split('test/')[1]
            subname = subpath.split('/')[0]
            fname = subpath.split('/')[1]

            sub_save_path = join(save_path, subname)
            if not isdir(sub_save_path):
                makedirs(sub_save_path)

            sio.savemat(join(sub_save_path, fname), test_output)


class DCE_supervised:
    def __init__(self, opt):
        self.device = opt.device
        self.disp_step = opt.disp_step
        self.lambda_cycle = opt.lambda_cycle
        self.lambda_tv = opt.lambda_tv
        self.lambda_l1 = opt.lambda_l1
        self.n_time = opt.n_time
        self.kinetic_model = opt.kinetic_model
        if self.kinetic_model == 'patlak':
            self.n_param = 2
        else:
            self.n_param = 3
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.n_epochs = opt.n_epochs
        self.n_epochs_decay = opt.n_epochs_decay
        self.continue_epoch = opt.continue_epoch
        self.test_epoch = opt.test_epoch
        self.lr = opt.lr
        self.init_type = opt.init_type
        self.init_gain = opt.init_gain
        self.save_epoch = opt.save_epoch
        self.save_path = opt.save_path
        self.experiment_name = opt.experiment_name
        self.scale_ktrans = opt.scale_ktrans
        self.scale_vp = opt.scale_vp
        self.scale_ve = opt.scale_ve
        self.opt = opt
        self.ckpt_dir = opt.ckpt_dir
        self.log_dir = opt.log_dir
        self.writer = SummaryWriter(self.log_dir)

        self.l1_loss = nn.L1Loss()
        self.loss_name = ['G/total_loss',
                          'G/param_loss']

        self.G_D2P = Dualnet(self.n_time, self.n_param, self.opt).to(self.device)
        if self.kinetic_model == 'patlak':
            self.G_P2D = patlak_model(self.opt).to(self.device)
        elif self.kinetic_model == 'etofts':
            self.G_P2D = etofts_model(self.opt).to(self.device)
        else:
            raise NotImplementedError('Model [%s] is not implemented' % self.kinetic_model)
        self.D_P = PatchGAN(self.n_param, self.opt).to(self.device)

        self.G_D2P_optim = torch.optim.Adam(self.G_D2P.parameters(), self.lr, betas=(self.beta1, self.beta2))
        self.G_D2P_scheduler = get_scheduler(self.G_D2P_optim, self.opt)

    def train(self, dataloader_train):
        if isfile(join(self.ckpt_dir, str(self.continue_epoch) + '.pth')):
            checkpoint = torch.load(join(self.ckpt_dir, str(self.continue_epoch) + '.pth'))
            self.G_D2P.load_state_dict(checkpoint['G_D2P_state_dict'])
            self.G_D2P_optim.load_state_dict(checkpoint['G_D2P_optim_state_dict'])
            self.G_D2P_scheduler.load_state_dict(checkpoint['G_D2P_scheduler_state_dict'])
            trained_epoch = checkpoint['epoch']
            print('Start from saved model - ' + str(trained_epoch))
        else:
            init_net(self.G_D2P, self.init_type, self.init_gain)
            trained_epoch = 0
            print('Start initially')

        losses = {name: Mean() for name in self.loss_name}
        dataset_train_len = len(dataloader_train)

        for epoch in tqdm(range(trained_epoch, self.n_epochs + self.n_epochs_decay), desc='Epoch', total=self.n_epochs + self.n_epochs_decay, initial=trained_epoch):
            for name in self.loss_name:
                losses[name].reset()
            disp_cnt = 0

            self.G_D2P.train()
            for step, (real_DCE, _, _, _, _, real_params, _, _, _, _) in enumerate(tqdm(dataloader_train)):
                real_DCE = real_DCE.to(self.device)
                real_params = real_params.to(self.device)

                fake_params, _ = self.G_D2P(real_DCE)

                G_param_loss = self.l1_loss(fake_params, real_params)
                G_total_loss = self.lambda_cycle * G_param_loss

                self.G_D2P_optim.zero_grad()
                G_total_loss.backward()
                self.G_D2P_optim.step()

                losses['G/param_loss'](G_param_loss.detach())
                losses['G/total_loss'](G_total_loss.detach())

                if step % self.disp_step == 0:
                    for name in self.loss_name:
                        self.writer.add_scalar('Step_' + name, losses[name].step(), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('Image/1_real_D', DCE2img(real_DCE, 0, 1), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('Image/2_fake_P', params2img(fake_params, self.scale_ktrans, self.scale_vp, self.scale_ve), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('Image/3_real_P', params2img(real_params, self.scale_ktrans, self.scale_vp, self.scale_ve), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    disp_cnt += 1

            for name in self.loss_name:
                self.writer.add_scalar('Epoch_' + name, losses[name].epoch(), epoch + 1)
            self.writer.add_scalar('Epoch/lr', self.G_D2P_scheduler.get_last_lr()[0], epoch + 1)

            self.G_D2P_scheduler.step()
            if (epoch + 1) % self.save_epoch == 0:
                torch.save({'epoch': epoch + 1, 'G_D2P_state_dict': self.G_D2P.state_dict(), 'G_D2P_optim_state_dict': self.G_D2P_optim.state_dict(),
                            'G_D2P_scheduler_state_dict': self.G_D2P_scheduler.state_dict()},
                           join(self.ckpt_dir, '{}'.format(epoch + 1) + '.pth'))

    def test(self, dataloader):
        checkpoint = torch.load(join(self.ckpt_dir, str(self.test_epoch) + '.pth'))
        self.G_D2P.load_state_dict(checkpoint['G_D2P_state_dict'])
        self.G_D2P.eval()

        save_path = join(self.save_path, self.experiment_name, 'test_unmask_{}'.format(self.test_epoch))
        if not isdir(save_path):
            makedirs(save_path)

        print('Start test!')

        for step, (real_DCE, _, _, _, mask_D, _, _, _, _, _) in enumerate(tqdm(dataloader)):
            real_DCE = real_DCE.to(self.device)
            mask_D = mask_D.to('cpu:0').detach().numpy()

            fake_params, _ = self.G_D2P(F.pad(real_DCE, [16, 16, 16, 16], "constant", 0))
            fake_params = np.squeeze(fake_params[:, :, 16:-16, 16:-16].to('cpu:0').detach().numpy())

            if self.kinetic_model == 'patlak':
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                test_output = {'Ktrans': Ktrans, 'vp': vp, 'mask': mask_D}
            else:
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                ve = fake_params[2, :, :] / self.scale_ve
                test_output = {'Ktrans': Ktrans, 'vp': vp, 've': ve, 'mask': mask_D}

            subpath = dataloader.flist[step].split('test/')[1]
            subname = subpath.split('/')[0]
            fname = subpath.split('/')[1]

            sub_save_path = join(save_path, subname)
            if not isdir(sub_save_path):
                makedirs(sub_save_path)

            sio.savemat(join(sub_save_path, fname), test_output)

    def test_simul(self, dataloader):
        checkpoint = torch.load(join(self.ckpt_dir, str(self.test_epoch) + '.pth'))
        self.G_D2P.load_state_dict(checkpoint['G_D2P_state_dict'])
        self.G_D2P.eval()

        save_path = join(self.save_path, self.experiment_name, 'test_simul_noise_{}'.format(self.test_epoch))
        if not isdir(save_path):
            makedirs(save_path)

        print('Start test!')

        for step, (_, _, _, _, _, real_params, Cp_P, S0_P, T1_P, mask_P) in enumerate(tqdm(dataloader)):
            real_params = real_params.to(self.device)
            Cp_P = Cp_P.to(self.device)
            S0_P = S0_P.to(self.device)
            T1_P = T1_P.to(self.device)
            mask_P = mask_P.to('cpu:0').detach().numpy()
            simul_DCE = self.G_P2D(real_params, Cp_P, S0_P, T1_P)
            simul_DCE += 0.02 * torch.randn(simul_DCE.size()).to(self.device)

            fake_params, _ = self.G_D2P(F.pad(simul_DCE, [16, 16, 16, 16], "constant", 0))
            fake_params = np.squeeze(fake_params[:, :, 16:-16, 16:-16].to('cpu:0').detach().numpy())

            if self.kinetic_model == 'patlak':
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                test_output = {'Ktrans': Ktrans, 'vp': vp, 'mask': mask_P}
            else:
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                ve = fake_params[2, :, :] / self.scale_ve
                test_output = {'Ktrans': Ktrans, 'vp': vp, 've': ve, 'mask': mask_P}

            subpath = dataloader.flist[step].split('test/')[1]
            subname = subpath.split('/')[0]
            fname = subpath.split('/')[1]

            sub_save_path = join(save_path, subname)
            if not isdir(sub_save_path):
                makedirs(sub_save_path)

            sio.savemat(join(sub_save_path, fname), test_output)


class DCE_supervised_physics:
    def __init__(self, opt):
        self.device = opt.device
        self.disp_step = opt.disp_step
        self.lambda_cycle = opt.lambda_cycle
        self.lambda_tv = opt.lambda_tv
        self.lambda_l1 = opt.lambda_l1
        self.n_time = opt.n_time
        self.kinetic_model = opt.kinetic_model
        if self.kinetic_model == 'patlak':
            self.n_param = 2
        else:
            self.n_param = 3
        self.beta1 = opt.beta1
        self.beta2 = opt.beta2
        self.n_epochs = opt.n_epochs
        self.n_epochs_decay = opt.n_epochs_decay
        self.continue_epoch = opt.continue_epoch
        self.test_epoch = opt.test_epoch
        self.lr = opt.lr
        self.init_type = opt.init_type
        self.init_gain = opt.init_gain
        self.save_epoch = opt.save_epoch
        self.save_path = opt.save_path
        self.experiment_name = opt.experiment_name
        self.scale_ktrans = opt.scale_ktrans
        self.scale_vp = opt.scale_vp
        self.scale_ve = opt.scale_ve
        self.opt = opt
        self.ckpt_dir = opt.ckpt_dir
        self.log_dir = opt.log_dir
        self.writer = SummaryWriter(self.log_dir)

        self.l1_loss = nn.L1Loss()
        self.loss_name = ['G/total_loss',
                          'G/param_loss',
                          'G/DCE_loss']

        self.G_D2P = Dualnet(self.n_time, self.n_param, self.opt).to(self.device)
        if self.kinetic_model == 'patlak':
            self.G_P2D = patlak_model(self.opt).to(self.device)
        elif self.kinetic_model == 'etofts':
            self.G_P2D = etofts_model(self.opt).to(self.device)
        else:
            raise NotImplementedError('Model [%s] is not implemented' % self.kinetic_model)

        self.G_D2P_optim = torch.optim.Adam(self.G_D2P.parameters(), self.lr, betas=(self.beta1, self.beta2))
        self.G_D2P_scheduler = get_scheduler(self.G_D2P_optim, self.opt)

    def train(self, dataloader_train):
        if isfile(join(self.ckpt_dir, str(self.continue_epoch) + '.pth')):
            checkpoint = torch.load(join(self.ckpt_dir, str(self.continue_epoch) + '.pth'))
            self.G_D2P.load_state_dict(checkpoint['G_D2P_state_dict'])
            self.G_D2P_optim.load_state_dict(checkpoint['G_D2P_optim_state_dict'])
            self.G_D2P_scheduler.load_state_dict(checkpoint['G_D2P_scheduler_state_dict'])
            trained_epoch = checkpoint['epoch']
            print('Start from saved model - ' + str(trained_epoch))
        else:
            init_net(self.G_D2P, self.init_type, self.init_gain)
            trained_epoch = 0
            print('Start initially')

        losses = {name: Mean() for name in self.loss_name}
        dataset_train_len = len(dataloader_train)

        for epoch in tqdm(range(trained_epoch, self.n_epochs + self.n_epochs_decay), desc='Epoch', total=self.n_epochs + self.n_epochs_decay, initial=trained_epoch):
            for name in self.loss_name:
                losses[name].reset()
            disp_cnt = 0

            self.G_D2P.train()
            for step, (real_DCE, Cp_D, S0_D, T1_D, _, real_params, _, _, _, _) in enumerate(tqdm(dataloader_train)):
                real_DCE = real_DCE.to(self.device)
                real_params = real_params.to(self.device)
                Cp_D = Cp_D.to(self.device)
                S0_D = S0_D.to(self.device)
                T1_D = T1_D.to(self.device)

                fake_params, _ = self.G_D2P(real_DCE)
                cycle_DCE = self.G_P2D(fake_params, Cp_D, S0_D, T1_D)

                G_param_loss = self.l1_loss(fake_params, real_params)
                G_DCE_loss = self.l1_loss(cycle_DCE, real_DCE)
                G_total_loss = self.lambda_cycle * (G_param_loss + G_DCE_loss)

                self.G_D2P_optim.zero_grad()
                G_total_loss.backward()
                self.G_D2P_optim.step()

                losses['G/param_loss'](G_param_loss.detach())
                losses['G/DCE_loss'](G_DCE_loss.detach())
                losses['G/total_loss'](G_total_loss.detach())

                if step % self.disp_step == 0:
                    for name in self.loss_name:
                        self.writer.add_scalar('Step_' + name, losses[name].step(), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('Image/1_real_D', DCE2img(real_DCE, 0, 1), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('Image/2_fake_P', params2img(fake_params, self.scale_ktrans, self.scale_vp, self.scale_ve), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('Image/3_cycle_D', DCE2img(cycle_DCE, 0, 1), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    self.writer.add_image('Image/4_real_P', params2img(real_params, self.scale_ktrans, self.scale_vp, self.scale_ve), epoch * ceil(dataset_train_len / self.disp_step) + disp_cnt)
                    disp_cnt += 1

            for name in self.loss_name:
                self.writer.add_scalar('Epoch_' + name, losses[name].epoch(), epoch + 1)
            self.writer.add_scalar('Epoch/lr', self.G_D2P_scheduler.get_last_lr()[0], epoch + 1)

            self.G_D2P_scheduler.step()
            if (epoch + 1) % self.save_epoch == 0:
                torch.save({'epoch': epoch + 1, 'G_D2P_state_dict': self.G_D2P.state_dict(), 'G_D2P_optim_state_dict': self.G_D2P_optim.state_dict(),
                            'G_D2P_scheduler_state_dict': self.G_D2P_scheduler.state_dict()},
                           join(self.ckpt_dir, '{}'.format(epoch + 1) + '.pth'))

    def test(self, dataloader):
        checkpoint = torch.load(join(self.ckpt_dir, str(self.test_epoch) + '.pth'))
        self.G_D2P.load_state_dict(checkpoint['G_D2P_state_dict'])
        self.G_D2P.eval()

        save_path = join(self.save_path, self.experiment_name, 'test_unmask_{}'.format(self.test_epoch))
        if not isdir(save_path):
            makedirs(save_path)

        print('Start test!')

        for step, (real_DCE, _, _, _, mask_D, _, _, _, _, _) in enumerate(tqdm(dataloader)):
            real_DCE = real_DCE.to(self.device)
            mask_D = mask_D.to('cpu:0').detach().numpy()

            fake_params, _ = self.G_D2P(F.pad(real_DCE, [16, 16, 16, 16], "constant", 0))
            fake_params = np.squeeze(fake_params[:, :, 16:-16, 16:-16].to('cpu:0').detach().numpy())
            if self.kinetic_model == 'patlak':
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                test_output = {'Ktrans': Ktrans, 'vp': vp, 'mask': mask_D}
            else:
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                ve = fake_params[2, :, :] / self.scale_ve
                test_output = {'Ktrans': Ktrans, 'vp': vp, 've': ve, 'mask': mask_D}

            subpath = dataloader.flist[step].split('test/')[1]
            subname = subpath.split('/')[0]
            fname = subpath.split('/')[1]

            sub_save_path = join(save_path, subname)
            if not isdir(sub_save_path):
                makedirs(sub_save_path)

            sio.savemat(join(sub_save_path, fname), test_output)

    def test_simul(self, dataloader):
        checkpoint = torch.load(join(self.ckpt_dir, str(self.test_epoch) + '.pth'))
        self.G_D2P.load_state_dict(checkpoint['G_D2P_state_dict'])
        self.G_D2P.eval()

        save_path = join(self.save_path, self.experiment_name, 'test_simul_noise_{}'.format(self.test_epoch))
        if not isdir(save_path):
            makedirs(save_path)

        print('Start test!')

        for step, (_, _, _, _, _, real_params, Cp_P, S0_P, T1_P, mask_P) in enumerate(tqdm(dataloader)):
            real_params = real_params.to(self.device)
            Cp_P = Cp_P.to(self.device)
            S0_P = S0_P.to(self.device)
            T1_P = T1_P.to(self.device)
            mask_P = mask_P.to('cpu:0').detach().numpy()
            simul_DCE = self.G_P2D(real_params, Cp_P, S0_P, T1_P)
            simul_DCE += 0.02 * torch.randn(simul_DCE.size()).to(self.device)

            fake_params, _ = self.G_D2P(F.pad(simul_DCE, [16, 16, 16, 16], "constant", 0))
            fake_params = np.squeeze(fake_params[:, :, 16:-16, 16:-16].to('cpu:0').detach().numpy())

            if self.kinetic_model == 'patlak':
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                test_output = {'Ktrans': Ktrans, 'vp': vp, 'mask': mask_P}
            else:
                Ktrans = fake_params[0, :, :] / self.scale_ktrans
                vp = fake_params[1, :, :] / self.scale_vp
                ve = fake_params[2, :, :] / self.scale_ve
                test_output = {'Ktrans': Ktrans, 'vp': vp, 've': ve, 'mask': mask_P}

            subpath = dataloader.flist[step].split('test/')[1]
            subname = subpath.split('/')[0]
            fname = subpath.split('/')[1]

            sub_save_path = join(save_path, subname)
            if not isdir(sub_save_path):
                makedirs(sub_save_path)

            sio.savemat(join(sub_save_path, fname), test_output)
