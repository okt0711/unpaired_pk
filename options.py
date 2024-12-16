import argparse
import os


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--FA', type=float, help='flip angle (degree)')
        self.parser.add_argument('--TR', type=float, help='repetition time (ms)')
        self.parser.add_argument('--act_type', type=str, default='lrelu', help='type of activation function')
        self.parser.add_argument('--augment', action='store_true', help='True if use data augmentation')
        self.parser.add_argument('--batch_size', type=int, help='batch size')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='moment for optimizer, 0 to 1')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='moment for optimizer, 0 to 1')
        self.parser.add_argument('--continue_epoch', type=int, default=1, help='continue training from continue_epoch')
        self.parser.add_argument('--data_root', type=str, help='path for data')
        self.parser.add_argument('--deltt', type=float, help='time interval between DCE image series (sec)')
        self.parser.add_argument('--disp_step', type=int, help='display step')
        self.parser.add_argument('--experiment_name', type=str, help='experiment name')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu_ids')
        self.parser.add_argument('--init_gain', type=float, default=1, help='gain of weight initialization')
        self.parser.add_argument('--init_type', type=str, default='xavier', help='the type of weight initialization method')
        self.parser.add_argument('--kinetic_model', type=str, help='which kinetic model')
        self.parser.add_argument('--lambda_adv', type=float, default=1, help='lambda for adversarial loss')
        self.parser.add_argument('--lambda_cycle', type=float, help='lambda for cycle-consistency loss')
        self.parser.add_argument('--lambda_cp', type=float, help='lambda for AIF loss')
        self.parser.add_argument('--lr', type=float, help='initial learning rate')
        self.parser.add_argument('--lr_decay_iters', type=int, help='lr decay iteration')
        self.parser.add_argument('--lr_policy', type=str, default='linear', help='the type of lr decay')
        self.parser.add_argument('--model', type=str, help='which model')
        self.parser.add_argument('--n_epochs', type=int, help='the number of epoch iteration')
        self.parser.add_argument('--n_epochs_decay', type=int, help='the number of decay epoch iteration')
        self.parser.add_argument('--n_time', type=int, help='the number of DCE image series')
        self.parser.add_argument('--ndf', type=int, help='the number of discriminator filters')
        self.parser.add_argument('--ngf', type=int, help='the number of generator filters')
        self.parser.add_argument('--norm_type', type=str, default='instance', help='normalization method (batch, instance, none)')
        self.parser.add_argument('--patch_size', type=int, help='the size of patch')
        self.parser.add_argument('--r1', type=float, help='relaxivity')
        self.parser.add_argument('--save_epoch', type=int, help='save model per N epochs')
        self.parser.add_argument('--save_path', type=str, help='path for saving results')
        self.parser.add_argument('--scale_ktrans', type=float, default=5, help='scale factor for Ktrans')
        self.parser.add_argument('--scale_ve', type=float, default=5/3, help='scale factor for ve')
        self.parser.add_argument('--scale_vp', type=float, default=10, help='scale factor for vp')
        self.parser.add_argument('--test_epoch', type=int, help='test using the model of test_epoch')
        self.parser.add_argument('--training', action='store_true', help='True if training mode')

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('---------- Options ----------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('------------ End ------------')

        expr_dir = os.path.join(self.opt.save_path, self.opt.experiment_name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('---------- Options ----------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('------------ End ------------')
        opt_file.close()
        return self.opt
