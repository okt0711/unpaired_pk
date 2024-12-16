import numpy as np
import torch
import random
from os import listdir
from os.path import join
from scipy import io as sio
from torch.utils.data import Dataset
from utils import masking


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class DCEDataset(Dataset):
    def __init__(self, opt, mode):
        self.mode = mode
        self.model = opt.model
        self.data_root = join(opt.data_root, self.mode)
        self.TR = opt.TR
        self.r1 = opt.r1
        self.FA = opt.FA

        self.flist_D = []
        sub_list = sorted(listdir(self.data_root))
        for aSub in sub_list:
            sub_root = join(self.data_root, aSub)
            for aImg in sorted(listdir(sub_root)):
                self.flist_D.append(join(sub_root, aImg))

        if self.mode == 'train':
            self.flist_P = []
            for aSub in sub_list:
                sub_root = join(self.data_root, aSub)
                for aImg in sorted(listdir(sub_root)):
                    self.flist_P.append(join(sub_root, aImg))
        else:
            self.flist_P = self.flist_D

        if 'cycle' in self.model and self.mode == 'train':
            random.shuffle(self.flist_D)
            random.shuffle(self.flist_P)

        self.kinetic_model = opt.kinetic_model
        self.scale_ktrans = opt.scale_ktrans
        self.scale_vp = opt.scale_vp
        self.scale_ve = opt.scale_ve
        self.patch_size = opt.patch_size

    def __len__(self):
        return len(min(self.flist_D, self.flist_P))

    def __getitem__(self, idx):
        file_D = sio.loadmat(self.flist_D[idx])
        file_P = sio.loadmat(self.flist_P[idx])

        ##############################
        mask_D = np.expand_dims(file_D['mask'], axis=0)
        DCE = np.clip(np.transpose(file_D['DCE'], [2, 0, 1]), 0, 3000)
        S0_D = np.mean(DCE[:4, :, :], axis=0, keepdims=True)
        if np.amax(S0_D) > 0:
            DCE = DCE / np.amax(S0_D)
            S0_D = S0_D / np.amax(S0_D)
        Cp_D = np.squeeze(file_D['AIF']) / (1 - 0.45)
        T1_D = file_D['T1']
        ##############################

        ##############################
        mask_P = np.expand_dims(file_P['mask'], axis=0)
        tmp_DCE = np.clip(np.transpose(file_P['DCE'], [2, 0, 1]), 0, 3000)
        S0_P = np.mean(tmp_DCE[:4, :, :], axis=0, keepdims=True)
        if np.amax(S0_P) > 0:
            S0_P = S0_P / np.amax(S0_P)
        Cp_P = np.squeeze(file_P['AIF']) / (1 - 0.45)
        T1_P = file_P['T1']

        if self.kinetic_model == 'patlak':
            Ktrans = np.expand_dims(file_P['Ktrans'], axis=0)
            vp = np.expand_dims(file_P['vp'], axis=0)
            mask_P, Ktrans, vp = masking(mask_P, [Ktrans, vp])
            Ktrans = Ktrans * self.scale_ktrans
            vp = vp * self.scale_vp
            params = np.concatenate([Ktrans, vp], axis=0)
        else:
            Ktrans = np.expand_dims(file_P['Ktrans'], axis=0)
            vp = np.expand_dims(file_P['vp'], axis=0)
            Kep = np.expand_dims(file_P['Kep'], axis=0)
            Kep[Kep == 0.0] = 1e-5
            ve = Ktrans / Kep
            ve[Kep == 0.0] = 0.0
            mask_P, Ktrans, vp, ve = masking(mask_P, [Ktrans, vp, ve])
            Ktrans = Ktrans * self.scale_ktrans
            vp = vp * self.scale_vp
            ve = ve * self.scale_ve
            params = np.concatenate([Ktrans, vp, ve], axis=0)

        if self.mode == 'train' and self.patch_size > 0:
            if 'supervised' in self.model:
                [DCE, S0_D, T1_D, mask_D, params, S0_P, T1_P, mask_P] = self.random_crop([DCE, S0_D, T1_D, mask_D, params, S0_P, T1_P, mask_P])
            else:
                [DCE, S0_D, T1_D, mask_D] = self.random_crop([DCE, S0_D, T1_D, mask_D])
                [params, S0_P, T1_P, mask_P] = self.random_crop([params, S0_P, T1_P, mask_P])

        DCE = torch.from_numpy(DCE)
        S0_D = torch.from_numpy(S0_D)
        T1_D = torch.from_numpy(T1_D).unsqueeze(0)
        mask_D = torch.from_numpy(mask_D)

        params = torch.from_numpy(params)
        S0_P = torch.from_numpy(S0_P)
        T1_P = torch.from_numpy(T1_P).unsqueeze(0)
        mask_P = torch.from_numpy(mask_P)

        if self.mode == 'train':
            if 'supervised' in self.model:
                DCE, S0_D, T1_D, mask_D, params, S0_P, T1_P, mask_P = self.augment_data([DCE, S0_D, T1_D, mask_D, params, S0_P, T1_P, mask_P])
            else:
                DCE, S0_D, T1_D, mask_D = self.augment_data([DCE, S0_D, T1_D, mask_D])
                params, S0_P, T1_P, mask_P = self.augment_data([params, S0_P, T1_P, mask_P])

        return DCE, Cp_D, S0_D, T1_D, mask_D, params, Cp_P, S0_P, T1_P, mask_P

    @staticmethod
    def augment_data(imgs):
        flip_h = []
        p_flip_h = random.random()
        if p_flip_h > 0.5:
            for i in range(len(imgs)):
                flip_h.append(torch.flip(imgs[i], [1]))
        else:
            flip_h = imgs

        flip_v = []
        p_flip_v = random.random()
        if p_flip_v > 0.5:
            for i in range(len(imgs)):
                flip_v.append(torch.flip(flip_h[i], [2]))
        else:
            flip_v = flip_h
        return flip_v

    def random_crop(self, imgs):
        size = np.shape(imgs[0])
        ys = random.randrange(int(size[-2] / 4), int(size[-2] / 4 * 3))
        xs = random.randrange(int(size[-1] / 4), int(size[-1] / 4 * 3))
        patches = [img[...,
                   ys - int(self.patch_size / 2):ys - int(self.patch_size / 2) + self.patch_size,
                   xs - int(self.patch_size / 2):xs - int(self.patch_size / 2) + self.patch_size] for img in imgs]
        return patches
