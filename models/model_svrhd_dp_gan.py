import os
import numpy as np
from math import log10
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from scipy.special import entr
import pdb

from networks import get_generator, get_discriminator
from networks.networks import gaussian_weights_init
from models.utils import LSGANLoss
from models.utils import AverageMeter, get_scheduler, psnr, mse, get_nonlinearity, compute_mi
from skimage.measure import compare_ssim as ssim
from . import losses


class RegModel(nn.Module):
    def __init__(self, opts):
        super(RegModel, self).__init__()

        self.networks = []
        self.optimizers = []
        self.loss_names = []

        # set default loss flags
        loss_flags = ("w_img_L1")
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G1 = get_generator(opts.net_G1, opts)
        self.net_G2 = get_generator(opts.net_G2, opts)
        self.net_D = get_discriminator(opts.net_D, opts)
        self.networks.append(self.net_G1)
        self.networks.append(self.net_G2)
        self.networks.append(self.net_D)

        self.opts = opts
        self.bidir = opts.bidir

        if self.is_train:
            if opts.image_loss == 'ncc':
                image_loss_func = losses.NCC().loss
            elif opts.image_loss == 'mse':
                image_loss_func = losses.MSE().loss
            else:
                raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % opts.image_loss)

            # 1st part: need two image loss functions if bidirectional
            self.weight_reg = opts.weight_reg
            if self.bidir:
                self.losses_reg_all = [image_loss_func, image_loss_func]
                self.weights_reg_all = [0.5, 0.5]
                self.losses_reg_names = ['loss_img_pos', 'loss_img_neg']
                self.loss_names = ['loss_img_pos', 'loss_img_neg']
            else:
                self.losses_reg_all = [image_loss_func]
                self.weights_reg_all = [1]
                self.losses_reg_names = ['loss_img_pos']
                self.loss_names = ['loss_img_pos']

            # prepare deformation loss
            self.losses_reg_all += [losses.Grad('l2', loss_mult=opts.int_downsize).loss]
            self.weights_reg_all += [opts.weight_deform]
            self.losses_reg_names += ['loss_deform']
            self.loss_names += ['loss_deform']

            # prepare intermediate denoising loss
            self.losses_reg_all += [image_loss_func]
            self.weights_reg_all += [0.2]
            self.losses_reg_names += ['loss_img_gate_dn']
            self.loss_names += ['loss_img_gate_dn']

            # 2nd part: prepare denoising GAN loss
            self.loss_dn_recon = losses.MSE().loss
            self.loss_dn_gan = LSGANLoss().cuda()
            self.weight_dn_recon = opts.weight_dn_recon
            self.loss_names += ['loss_D', 'loss_G_GAN', 'loss_G_recon']

            # prepare optimizer
            params_G = list(self.net_G1.parameters()) + list(self.net_G2.parameters())
            self.optimizer_G = torch.optim.Adam(params_G,
                                                lr=opts.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(),
                                                lr=opts.lr, betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.vol_G1LD = data['vol_G1LD'].to(self.device).float()
        self.vol_G2LD = data['vol_G2LD'].to(self.device).float()
        self.vol_G3LD = data['vol_G3LD'].to(self.device).float()
        self.vol_G4LD = data['vol_G4LD'].to(self.device).float()
        self.vol_G5LD = data['vol_G5LD'].to(self.device).float()
        self.vol_G6LD = data['vol_G6LD'].to(self.device).float()
        self.vol_G1HD = data['vol_G1HD'].to(self.device).float()
        self.vol_G2HD = data['vol_G2HD'].to(self.device).float()
        self.vol_G3HD = data['vol_G3HD'].to(self.device).float()
        self.vol_G4HD = data['vol_G4HD'].to(self.device).float()
        self.vol_G5HD = data['vol_G5HD'].to(self.device).float()
        self.vol_G6HD = data['vol_G6HD'].to(self.device).float()
        self.vols_G12356LD = data['vols_G12356LD'].to(self.device).float()
        self.vols_G12356HD = data['vols_G12356HD'].to(self.device).float()
        self.vol_zeros = data['vol_zeros'].to(self.device).float()

        self.inp_LD_src = self.vols_G12356LD
        self.inp_LD_tgt = self.vol_G4LD
        self.inp_HD_src = self.vols_G12356HD
        self.inp_HD_tgt = self.vol_G4HD

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        self.inp_LD_src.requires_grad_(True)
        self.inp_LD_tgt.requires_grad_(True)
        self.inp_HD_src.requires_grad_(True)
        self.inp_HD_tgt.requires_grad_(True)

        if self.bidir:
            self.warped_HD_src, self.warped_HD_tgt, \
            self.warped_LD_src, self.warped_LD_tgt, \
            self.T_preint, self.T, \
            self.pred_HD_src, self.pred_HD_tgt \
                = self.net_G1(self.inp_LD_src, self.inp_LD_tgt, self.inp_HD_src, self.inp_HD_tgt)
        else:
            self.warped_HD_src, \
            self.warped_LD_src, \
            self.T_preint, self.T, \
            self.pred_HD_src, self.pred_HD_tgt \
                = self.net_G1(self.inp_LD_src, self.inp_LD_tgt, self.inp_HD_src, self.inp_HD_tgt)

        self.avg_LD = self.inp_LD_tgt + self.warped_LD_src[:, 0:1, :, :, :, :] + \
                      self.warped_LD_src[:, 1:2, :, :, :, :] + self.warped_LD_src[:, 2:3, :, :, :, :] + \
                      self.warped_LD_src[:, 3:4, :, :, :, :] + self.warped_LD_src[:, 4:5, :, :, :, :]

        self.pred_HD_final = self.net_G2(self.avg_LD[:, 0, :, :, :, :])

    def optimize(self):
        self.net_D.zero_grad()

        # fake
        if self.bidir:
            self.warped_HD_src, self.warped_HD_tgt, \
            self.warped_LD_src, self.warped_LD_tgt, \
            self.T_preint, self.T, \
            self.pred_HD_src, self.pred_HD_tgt \
                = self.net_G1(self.inp_LD_src, self.inp_LD_tgt, self.inp_HD_src, self.inp_HD_tgt)
        else:
            self.warped_HD_src, \
            self.warped_LD_src, \
            self.T_preint, self.T, \
            self.pred_HD_src, self.pred_HD_tgt \
                = self.net_G1(self.inp_LD_src, self.inp_LD_tgt, self.inp_HD_src, self.inp_HD_tgt)

        self.avg_LD = self.inp_LD_tgt + self.warped_LD_src[:, 0:1, :, :, :, :] + \
                      self.warped_LD_src[:, 1:2, :, :, :, :] + self.warped_LD_src[:, 2:3, :, :, :, :] + \
                      self.warped_LD_src[:, 3:4, :, :, :, :] + self.warped_LD_src[:, 4:5, :, :, :, :]

        self.pred_HD_final = self.net_G2(self.avg_LD[:, 0, :, :, :, :])

        pred_fake = self.net_D(self.pred_HD_final.detach())
        loss_D_fake = self.loss_dn_gan(pred_fake, target_is_real=False)

        # real
        pred_real = self.net_D(self.inp_HD_tgt[:, 0, :, :, :, :])
        loss_D_real = self.loss_dn_gan(pred_real, target_is_real=True)
        self.loss_D = (loss_D_fake + loss_D_real) * 0.5
        self.loss_D.backward()
        self.optimizer_D.step()

        # adv
        self.net_G1.zero_grad()
        self.net_G2.zero_grad()

        if self.bidir:
            self.warped_HD_src, self.warped_HD_tgt, \
            self.warped_LD_src, self.warped_LD_tgt, \
            self.T_preint, self.T, \
            self.pred_HD_src, self.pred_HD_tgt \
                = self.net_G1(self.inp_LD_src, self.inp_LD_tgt, self.inp_HD_src, self.inp_HD_tgt)
        else:
            self.warped_HD_src, \
            self.warped_LD_src, \
            self.T_preint, self.T, \
            self.pred_HD_src, self.pred_HD_tgt \
                = self.net_G1(self.inp_LD_src, self.inp_LD_tgt, self.inp_HD_src, self.inp_HD_tgt)

        self.avg_LD = self.inp_LD_tgt + self.warped_LD_src[:, 0:1, :, :, :, :] + \
                      self.warped_LD_src[:, 1:2, :, :, :, :] + self.warped_LD_src[:, 2:3, :, :, :, :] + \
                      self.warped_LD_src[:, 3:4, :, :, :, :] + self.warped_LD_src[:, 4:5, :, :, :, :]

        self.pred_HD_final = self.net_G2(self.avg_LD[:, 0, :, :, :, :])

        pred_fake = self.net_D(self.pred_HD_final)
        self.loss_G_GAN = self.loss_dn_gan(pred_fake, target_is_real=True)

        self.loss_G_recon = self.loss_dn_recon(self.pred_HD_final, self.inp_HD_tgt)

        self.loss_G_reg = 0
        self.loss_G_reg_list = []
        for n, loss_function in enumerate(self.losses_reg_all):
            if self.bidir:
                if n == 0:
                    curr_loss = loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 0, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 1, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 2, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 3, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 4, :, :, :, :]) * self.weights_reg_all[n]
                    self.loss_img_pos = curr_loss

                if n == 1:
                    curr_loss = loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 0, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 1, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 2, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 3, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 4, :, :, :, :]) * self.weights_reg_all[n]
                    self.loss_img_neg = curr_loss

                if n == 2:
                    curr_loss = loss_function(self.vol_zeros[:, 0, :, :, :, :], self.T_preint[:, 0, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_zeros[:, 0, :, :, :, :], self.T_preint[:, 1, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_zeros[:, 0, :, :, :, :], self.T_preint[:, 2, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_zeros[:, 0, :, :, :, :], self.T_preint[:, 3, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_zeros[:, 0, :, :, :, :], self.T_preint[:, 4, :, :, :, :]) * self.weights_reg_all[n]
                    self.loss_deform = curr_loss

                if n == 3:
                    curr_loss1 = loss_function(self.vol_G1HD[:, 0, :, :, :, :], self.pred_HD_src[:, 0, :, :, :, :]) * self.weights_reg_all[n] + \
                                 loss_function(self.vol_G2HD[:, 0, :, :, :, :], self.pred_HD_src[:, 1, :, :, :, :]) * self.weights_reg_all[n] + \
                                 loss_function(self.vol_G3HD[:, 0, :, :, :, :], self.pred_HD_src[:, 2, :, :, :, :]) * self.weights_reg_all[n] + \
                                 loss_function(self.vol_G5HD[:, 0, :, :, :, :], self.pred_HD_src[:, 3, :, :, :, :]) * self.weights_reg_all[n] + \
                                 loss_function(self.vol_G6HD[:, 0, :, :, :, :], self.pred_HD_src[:, 4, :, :, :, :]) * self.weights_reg_all[n]
                    curr_loss2 = loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.pred_HD_tgt[:, 0, :, :, :, :]) * self.weights_reg_all[n]
                    curr_loss = curr_loss1 + curr_loss2
                    self.loss_img_gate_dn = curr_loss

                self.loss_G_reg_list.append('%.6f' % curr_loss.item())
                self.loss_G_reg += curr_loss

            if not self.bidir:
                if n == 0:
                    curr_loss = loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 0, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 1, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 2, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 3, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.warped_HD_src[:, 4, :, :, :, :]) * self.weights_reg_all[n]
                    self.loss_img_pos = curr_loss

                if n == 1:
                    curr_loss = loss_function(self.vol_zeros[:, 0, :, :, :, :], self.T_preint[:, 0, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_zeros[:, 0, :, :, :, :], self.T_preint[:, 1, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_zeros[:, 0, :, :, :, :], self.T_preint[:, 2, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_zeros[:, 0, :, :, :, :], self.T_preint[:, 3, :, :, :, :]) * self.weights_reg_all[n] + \
                                loss_function(self.vol_zeros[:, 0, :, :, :, :], self.T_preint[:, 4, :, :, :, :]) * self.weights_reg_all[n]
                    self.loss_deform = curr_loss

                if n == 2:
                    curr_loss1 = loss_function(self.vol_G1HD[:, 0, :, :, :, :], self.pred_HD_src[:, 0, :, :, :, :]) * self.weights_reg_all[n] + \
                                 loss_function(self.vol_G2HD[:, 0, :, :, :, :], self.pred_HD_src[:, 1, :, :, :, :]) * self.weights_reg_all[n] + \
                                 loss_function(self.vol_G3HD[:, 0, :, :, :, :], self.pred_HD_src[:, 2, :, :, :, :]) * self.weights_reg_all[n] + \
                                 loss_function(self.vol_G5HD[:, 0, :, :, :, :], self.pred_HD_src[:, 3, :, :, :, :]) * self.weights_reg_all[n] + \
                                 loss_function(self.vol_G6HD[:, 0, :, :, :, :], self.pred_HD_src[:, 4, :, :, :, :]) * self.weights_reg_all[n]
                    curr_loss2 = loss_function(self.vol_G4HD[:, 0, :, :, :, :], self.pred_HD_tgt[:, 0, :, :, :, :]) * self.weights_reg_all[n]
                    curr_loss = curr_loss1 + curr_loss2
                    self.loss_img_gate_dn = curr_loss

                self.loss_G_reg_list.append('%.6f' % curr_loss.item())
                self.loss_G_reg += curr_loss

        self.loss_G = self.loss_G_GAN + self.loss_G_recon * self.weight_dn_recon + self.loss_G_reg * self.weight_reg
        self.loss_G.backward()
        self.optimizer_G.step()

    @property
    def loss_summary(self):
        message = 'loss_G_reg: %.6f  (%s) ' % (self.loss_G_reg.item(), ', '.join(self.loss_G_reg_list))
        message += 'loss_D: {:4f}, loss_G(GAN): {:4f}, loss_G(recon): {:4f}'.format(self.loss_D.item(),
                                                                                    self.loss_G_GAN.item(),
                                                                                    self.loss_G_recon.item())
        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):

        state = {}
        state['net_G1'] = self.net_G1.module.state_dict()
        state['net_G2'] = self.net_G2.module.state_dict()
        state['net_D'] = self.net_D.module.state_dict()
        state['opt_G'] = self.optimizer_G.state_dict()
        state['opt_D'] = self.optimizer_D.state_dict()
        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file)

        self.net_G1.module.load_state_dict(checkpoint['net_G1'])
        self.net_G2.module.load_state_dict(checkpoint['net_G2'])
        self.net_D.module.load_state_dict(checkpoint['net_D'])
        if train:
            self.optimizer_G.load_state_dict(checkpoint['opt_G'])
            self.optimizer_D.load_state_dict(checkpoint['opt_D'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)
        avg_mi_G1 = AverageMeter()
        avg_mi_G2 = AverageMeter()
        avg_mi_G3 = AverageMeter()
        avg_mi_G5 = AverageMeter()
        avg_mi_G6 = AverageMeter()
        avg_mi_Gmean = AverageMeter()
        avg_mse_DN = AverageMeter()

        warped_HD_src_all = []
        warped_LD_src_all = []
        Tfield_all = []
        inp_HD_tgt_all = []
        inp_LD_tgt_all = []
        inp_HD_src_all = []
        inp_LD_src_all = []
        pred_HD_all = []

        for data in val_bar:
            self.set_input(data)
            self.forward()

            mi_G1 = compute_mi(self.warped_HD_src[0, 0, 0, :, :, :].cpu().numpy(), self.inp_HD_tgt[0, 0, 0, :, :, :].cpu().numpy())
            mi_G2 = compute_mi(self.warped_HD_src[0, 1, 0, :, :, :].cpu().numpy(), self.inp_HD_tgt[0, 0, 0, :, :, :].cpu().numpy())
            mi_G3 = compute_mi(self.warped_HD_src[0, 2, 0, :, :, :].cpu().numpy(), self.inp_HD_tgt[0, 0, 0, :, :, :].cpu().numpy())
            mi_G5 = compute_mi(self.warped_HD_src[0, 3, 0, :, :, :].cpu().numpy(), self.inp_HD_tgt[0, 0, 0, :, :, :].cpu().numpy())
            mi_G6 = compute_mi(self.warped_HD_src[0, 4, 0, :, :, :].cpu().numpy(), self.inp_HD_tgt[0, 0, 0, :, :, :].cpu().numpy())
            mi_Gmean = 1/5 * (mi_G1 + mi_G2 + mi_G3 + mi_G5 + mi_G6)
            mse_DN = mse(self.pred_HD_final[:,0,:,:,:], self.inp_HD_tgt[:,0,0,:,:,:])

            avg_mi_G1.update(mi_G1)
            avg_mi_G2.update(mi_G2)
            avg_mi_G3.update(mi_G3)
            avg_mi_G5.update(mi_G5)
            avg_mi_G6.update(mi_G6)
            avg_mi_Gmean.update(mi_Gmean)
            avg_mse_DN.update(mse_DN)

            warped_HD_src_all.append(self.warped_HD_src[:, :, 0, :, :, :].cpu())
            warped_LD_src_all.append(self.warped_LD_src[:, :, 0, :, :, :].cpu())
            Tfield_all.append(self.T.cpu())
            inp_HD_tgt_all.append(self.inp_HD_tgt[:, :, 0, :, :, :].cpu())
            inp_LD_tgt_all.append(self.inp_LD_tgt[:, :, 0, :, :, :].cpu())
            inp_HD_src_all.append(self.inp_HD_src[:, :, 0, :, :, :].cpu())
            inp_LD_src_all.append(self.inp_LD_src[:, :, 0, :, :, :].cpu())
            pred_HD_all.append(self.pred_HD_final.cpu())

            message = 'MI-Gmean: {:4f} ' \
                      'MI-G1: {:4f} MI-G2: {:4f} MI-G3: {:4f} MI-G5: {:4f} MI-G6: {:4f} ' \
                      'MSE-dn: {:4f}'.format(avg_mi_Gmean.avg,
                                             avg_mi_G1.avg, avg_mi_G2.avg, avg_mi_G3.avg, avg_mi_G5.avg, avg_mi_G6.avg,
                                             avg_mse_DN.avg)
            val_bar.set_description(desc=message)

        self.mi_G1 = avg_mi_G1.avg
        self.mi_G2 = avg_mi_G2.avg
        self.mi_G3 = avg_mi_G3.avg
        self.mi_G5 = avg_mi_G5.avg
        self.mi_G6 = avg_mi_G6.avg
        self.mi_Gmean = avg_mi_Gmean.avg
        self.mse_DN = avg_mse_DN.avg

        self.results = {}
        self.results['warped_HD_src_all'] = torch.stack(warped_HD_src_all).squeeze().numpy()
        self.results['warped_LD_src_all'] = torch.stack(warped_LD_src_all).squeeze().numpy()
        self.results['Tfield_all'] = torch.stack(Tfield_all).squeeze().numpy()
        self.results['inp_HD_tgt_all'] = torch.stack(inp_HD_tgt_all).squeeze().numpy()
        self.results['inp_LD_tgt_all'] = torch.stack(inp_LD_tgt_all).squeeze().numpy()
        self.results['inp_HD_src_all'] = torch.stack(inp_HD_src_all).squeeze().numpy()
        self.results['inp_LD_src_all'] = torch.stack(inp_LD_src_all).squeeze().numpy()
        self.results['pred_HD_all'] = torch.stack(pred_HD_all).squeeze().numpy()