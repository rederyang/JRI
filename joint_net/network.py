import os
import time

import torch
from torch import nn

import matplotlib.pyplot as plt

from mri_tools import *
from utils import *
from joint_net.base_network import BaseModel


class JointReconInterModel(BaseModel):
    def build(self):
        self.kspace_criterion = nn.L1Loss()
        self.imgspace_criterion = nn.MSELoss()

    def attach_meters(self):
        pass

    def attach_subnetworks(self, rec_model=None, inter_model=None):
        if rec_model is not None and inter_model is not None:
            self.rec_model = rec_model
            self.inter_model = inter_model
        else:
            raise NotImplementedError("You must provide pretrained rec_model and inter_model")

        self.rec_model_optimizer = torch.optim.Adam(list(self.rec_model.parameters()), lr=self.args.rec_lr)
        self.inter_model_optimizer = torch.optim.Adam(list(self.inter_model.parameters()), lr=self.args.inter_lr)

    def set_input(self, mode, data_batch):
        """
        all shape == [bs, 2, x, y]
        """
        data_batch = [item.to(self.rank, non_blocking=True) for item in data_batch]

        if mode == 'train':
            self.slice_1, self.slice_3, self.slice_5, self.mask_omega = data_batch

            self.slice_1 = torch.view_as_real(self.slice_1[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            self.slice_3 = torch.view_as_real(self.slice_3[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            self.slice_5 = torch.view_as_real(self.slice_5[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            self.mask_omega = self.mask_omega.to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()

            self.slice_1_k_omega = fft2_tensor(self.slice_1) * self.mask_omega
            self.slice_3_k_omega = fft2_tensor(self.slice_3) * self.mask_omega
            self.slice_5_k_omega = fft2_tensor(self.slice_5) * self.mask_omega

            self.slice_1_img_omega = ifft2_tensor(self.slice_1_k_omega)
            self.slice_3_img_omega = ifft2_tensor(self.slice_3_k_omega)
            self.slice_5_img_omega = ifft2_tensor(self.slice_5_k_omega)

            self.slice_1_mask_omega = self.mask_omega
            self.slice_2_mask_omega = self.mask_omega
            self.slice_3_mask_omega = self.mask_omega

        else:
            self.slice_1, self.slice_2, self.slice_3, self.mask_omega = data_batch

            self.slice_1 = torch.view_as_real(self.slice_1[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            self.slice_2 = torch.view_as_real(self.slice_2[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            self.slice_3 = torch.view_as_real(self.slice_3[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            self.mask_omega = self.mask_omega.to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()

            self.slice_1_k_omega = fft2_tensor(self.slice_1) * self.mask_omega
            self.slice_2_k_omega = fft2_tensor(self.slice_2) * self.mask_omega
            self.slice_3_k_omega = fft2_tensor(self.slice_3) * self.mask_omega
            self.slice_1_img_omega = ifft2_tensor(self.slice_1_k_omega)
            self.slice_2_img_omega = ifft2_tensor(self.slice_2_k_omega)
            self.slice_3_img_omega = ifft2_tensor(self.slice_3_k_omega)

    def train_forward(self):
        loss = 0.

        # 1st order recon
        self.slice_1_img_rec = self.rec_model.recon(self.slice_1_img_omega, self.slice_1_k_omega, self.mask_omega)
        self.slice_3_img_rec = self.rec_model.recon(self.slice_3_img_omega, self.slice_3_k_omega, self.mask_omega)
        self.slice_5_img_rec = self.rec_model.recon(self.slice_5_img_omega, self.slice_5_k_omega, self.mask_omega)

        # to real, abs
        self.slice_1_img_rec = torch.abs(real2complex_tensor(self.slice_1_img_rec))
        self.slice_3_img_rec = torch.abs(real2complex_tensor(self.slice_3_img_rec))
        self.slice_5_img_rec = torch.abs(real2complex_tensor(self.slice_5_img_rec))

        # 1st order inter
        self.slice_2_img_inter = self.inter_model.recon(self.slice_1_img_rec, self.slice_3_img_rec)
        self.slice_4_img_inter = self.inter_model.recon(self.slice_3_img_rec, self.slice_5_img_rec)

        # # to complex, to real form
        # self.slice_2_img_inter = complex2real_tensor(self.slice_2_img_inter.type(torch.complex64))
        # self.slice_4_img_inter = complex2real_tensor(self.slice_4_img_inter.type(torch.complex64))
        #
        # # 2nd order under-sampling
        # self.slice_2_k_omega = fft2_tensor(self.slice_2_img_inter) * self.mask_omega
        # self.slice_4_k_omega = fft2_tensor(self.slice_4_img_inter) * self.mask_omega
        # self.slice_2_img_omega = ifft2_tensor(self.slice_2_k_omega)
        # self.slice_4_img_omega = ifft2_tensor(self.slice_4_k_omega)
        #
        # # 2nd order recon
        # self.slice_2_img_rec = self.rec_model.recon(self.slice_2_img_omega, self.slice_2_img_omega, self.mask_omega)
        # self.slice_4_img_rec = self.rec_model.recon(self.slice_4_img_omega, self.slice_4_img_omega, self.mask_omega)
        #
        # # to real, abs
        # self.slice_2_img_rec = torch.abs(real2complex_tensor(self.slice_2_img_rec))
        # self.slice_4_img_rec = torch.abs(real2complex_tensor(self.slice_4_img_rec))
        #
        # # 2nd order inter
        # self.slice_3_img_inter = self.inter_model.recon(self.slice_2_img_rec, self.slice_4_img_rec)

        # 2nd order inter
        self.slice_3_img_inter = self.inter_model.recon(self.slice_2_img_inter, self.slice_4_img_inter)

        # to complex, to real form
        self.slice_3_img_inter = complex2real_tensor(self.slice_3_img_inter.type(torch.complex64))

        alpha1 = 1
        alpha2 = 1

        slice_3_k_diff_inter_omega = (fft2_tensor(self.slice_3_img_inter) - self.slice_3_k_omega) * self.mask_omega
        loss_omega = self.kspace_criterion(slice_3_k_diff_inter_omega, torch.zeros_like(slice_3_k_diff_inter_omega))
        loss_omega = loss_omega * alpha1

        slice_3_img_diff_rec_inter = self.slice_3_img_rec - self.slice_3_img_inter
        loss_img = self.imgspace_criterion(slice_3_img_diff_rec_inter, torch.zeros_like(slice_3_img_diff_rec_inter))
        loss_img = loss_img * alpha2

        loss = loss_omega + loss_img

        self.to_meters('train_loss_omega', loss_omega.item())
        self.to_meters('train_loss_img', loss_img.item())
        self.to_meters('train_loss', loss.item())

        return loss

    def update(self):
        loss = self.train_forward()
        self.inter_model_optimizer.zero_grad()
        self.rec_model_optimizer.zero_grad()
        loss.backward()
        self.inter_model_optimizer.step()
        self.rec_model_optimizer.step()

    def inference(self):

        self.slice_1_img_rec = self.rec_model.recon(self.slice_1_img_omega, self.slice_1_k_omega, self.mask_omega)
        self.slice_3_img_rec = self.rec_model.recon(self.slice_3_img_omega, self.slice_3_k_omega, self.mask_omega)

        self.slice_img_rec = torch.cat([self.slice_1_img_rec, self.slice_3_img_rec], dim=0)
        self.slice_img_rec_ref = torch.cat([self.slice_1, self.slice_3], dim=0)

        # to real, abs
        self.slice_1_img_rec = torch.abs(real2complex_tensor(self.slice_1_img_rec))
        self.slice_3_img_rec = torch.abs(real2complex_tensor(self.slice_3_img_rec))

        self.slice_2_img_inter = self.inter_model.recon(self.slice_1_img_rec, self.slice_3_img_rec)

        # to complex, to real form
        self.slice_2_img_inter = complex2real_tensor(self.slice_2_img_inter.type(torch.complex64))

    def post_evaluation(self):
        # for rec-inter slices
        # get magnitude images
        img_ref = torch.abs(torch.view_as_complex(self.slice_2.permute(0, 2, 3, 1).contiguous()))
        img_output = torch.abs(torch.view_as_complex(self.slice_2_img_inter.permute(0, 2, 3, 1).contiguous()))
        img_diff = img_output - img_ref

        # calculate metrics
        psnr =  psnr_slice(img_ref, img_output)
        ssim = ssim_slice(img_ref, img_output)

        self.to_meters('val_inter_image_psnr', psnr)
        self.to_meters('val_inter_image_ssim', ssim)

        # for rec slices
        # get magnitude images
        img_rec_ref = torch.abs(torch.view_as_complex(self.slice_img_rec_ref.permute(0, 2, 3, 1).contiguous()))
        img_rec_output = torch.abs(torch.view_as_complex(self.slice_img_rec.permute(0, 2, 3, 1).contiguous()))
        img_rec_diff = img_rec_output - img_rec_ref

        # calculate metrics
        psnr =  psnr_slice(img_rec_ref, img_rec_output)
        ssim = ssim_slice(img_rec_ref, img_rec_output)

        self.to_meters('val_rec_image_psnr', psnr)
        self.to_meters('val_rec_image_ssim', ssim)

        if self.save_test_vis:
            if not hasattr(self, 'cnt'):
                self.cnt = 0
            else:
                self.cnt += 1
            # rec-inter
            plt.imsave(os.path.join(self.args.output_path, f'img_ref_{self.cnt}.png'), img_ref.detach().cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_output_{self.cnt}.png'), img_output.detach().cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_diff_{self.cnt}.png'), img_diff.detach().cpu()[0], cmap='bwr', vmin=-0.3, vmax=0.3)
            # inter
            plt.imsave(os.path.join(self.args.output_path, f'img_rec_ref_{self.cnt}.png'), img_rec_ref.detach().cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_rec_output_{self.cnt}.png'), img_rec_output.detach().cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_rec_diff_{self.cnt}.png'), img_rec_diff.detach().cpu()[0], cmap='bwr', vmin=-0.3, vmax=0.3)


    def run_one_epoch(self, mode, dataloader):

        assert mode in ['train', 'val', 'test']

        self.reset_meters()

        tik = time.time()
        for iter_num, data_batch in enumerate(dataloader):
            self.set_input(mode, data_batch)
            if mode == 'train':
                self.update()
                # no post_evaluation
            else:
                self.inference()
                self.post_evaluation()
        tok = time.time()

        log = dict()
        log['epoch'] = self.epoch
        log['time'] = tok - tik
        if mode == 'train':
            log['rec_lr'] = self.rec_model_optimizer.param_groups[0]['lr']
            log['inter_lr'] = self.inter_model_optimizer.param_groups[0]['lr']
        log.update(self.summarize_meters())

        return log


JRI = JointReconInterModel
