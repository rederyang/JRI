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

        self.rec_sup_weight = self.args.rec_sup_weight
        self.rec_unsup_weight = self.args.rec_unsup_weight
        self.inter_sup_weight = self.args.inter_sup_weight
        self.inter_unsup_weight = self.args.inter_unsup_weight

    def attach_meters(self):
        pass

    def attach_subnetworks(self, rec_model=None, rec_model_2=None, inter_model=None):
        self.optimizers = list()
        if rec_model is not None and inter_model is not None:
            self.rec_model = rec_model
            self.inter_model = inter_model
            self.optimizers.append(torch.optim.Adam(list(self.rec_model.parameters()), lr=self.args.rec_lr))
            self.optimizers.append(torch.optim.Adam(list(self.inter_model.parameters()), lr=self.args.inter_lr))
            if rec_model_2 is not None:
                self.rec_model_2 = rec_model_2
                self.optimizers.append(torch.optim.Adam(list(self.rec_model_2.parameters()), lr=self.args.rec_lr_2))
        else:
            raise NotImplementedError("You must provide pretrained rec_model and inter_model")


    def set_input(self, mode, data_batch):
        """
        all shape == [bs, 2, x, y]
        """
        if mode == 'train':  # data_batch is a tuple
            sup_sample = dict()
            slice_1, slice_2, slice_3, mask_omega = data_batch[0]
            slice_1 = torch.view_as_real(slice_1[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            slice_2 = torch.view_as_real(slice_2[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            slice_3 = torch.view_as_real(slice_3[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            mask_omega = mask_omega.to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()

            slice_2_k = fft2_tensor(slice_2)
            slice_2_k_omega = fft2_tensor(slice_2) * mask_omega
            slice_2_img_omega = ifft2_tensor(slice_2_k_omega)

            sup_sample['slice_1'] = slice_1
            sup_sample['slice_2'] = slice_2
            sup_sample['slice_3'] = slice_3
            sup_sample['slice_2_k'] = slice_2_k
            sup_sample['mask_omega'] = mask_omega
            sup_sample['slice_2_k_omega'] = slice_2_k_omega
            sup_sample['slice_2_img_omega'] = slice_2_img_omega
            self.sup_sample = sup_sample

            unsup_sample = dict()
            slice_1, slice_3, slice_5, mask_omega = data_batch[1]
            slice_1 = torch.view_as_real(slice_1[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            slice_3 = torch.view_as_real(slice_3[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            slice_5 = torch.view_as_real(slice_5[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            mask_omega = mask_omega.to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()

            slice_1_k = fft2_tensor(slice_1)
            slice_3_k = fft2_tensor(slice_3)
            slice_5_k = fft2_tensor(slice_5)
            slice_1_k_omega = fft2_tensor(slice_1) * mask_omega
            slice_3_k_omega = fft2_tensor(slice_3) * mask_omega
            slice_5_k_omega = fft2_tensor(slice_5) * mask_omega

            slice_1_img_omega = ifft2_tensor(slice_1_k_omega)
            slice_3_img_omega = ifft2_tensor(slice_3_k_omega)
            slice_5_img_omega = ifft2_tensor(slice_5_k_omega)

            unsup_sample['slice_1'] = slice_1
            unsup_sample['slice_3'] = slice_3
            unsup_sample['slice_5'] = slice_5
            unsup_sample['slice_1_k'] = slice_1_k
            unsup_sample['slice_3_k'] = slice_3_k
            unsup_sample['slice_5_k'] = slice_5_k
            unsup_sample['mask_omega'] = mask_omega
            unsup_sample['slice_1_k_omega'] = slice_1_k_omega
            unsup_sample['slice_3_k_omega'] = slice_3_k_omega
            unsup_sample['slice_5_k_omega'] = slice_5_k_omega
            unsup_sample['slice_1_img_omega'] = slice_1_img_omega
            unsup_sample['slice_3_img_omega'] = slice_3_img_omega
            unsup_sample['slice_5_img_omega'] = slice_5_img_omega
            self.unsup_sample = unsup_sample

        else:
            inf_sample = dict()
            slice_1, slice_3, slice_2, mask_omega = data_batch
            slice_1 = torch.view_as_real(slice_1[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            slice_2 = torch.view_as_real(slice_2[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            slice_3 = torch.view_as_real(slice_3[:, 0].to(self.rank, non_blocking=True)).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
            mask_omega = mask_omega.to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()

            slice_2_k = fft2_tensor(slice_2)
            slice_2_k_omega = fft2_tensor(slice_2) * mask_omega
            slice_2_img_omega = ifft2_tensor(slice_2_k_omega)

            inf_sample['slice_1'] = slice_1
            inf_sample['slice_2'] = slice_2
            inf_sample['slice_3'] = slice_3
            inf_sample['slice_2_k'] = slice_2_k
            inf_sample['mask_omega'] = mask_omega
            inf_sample['slice_2_k_omega'] = slice_2_k_omega
            inf_sample['slice_2_img_omega'] = slice_2_img_omega
            self.inf_sample = inf_sample

    def sup_rec_forward(self):
        # sup recon (only rec the second slice in the triplet slices)
        slice_2_img_rec, slice_2_rec_loss = self.rec_model.forward_get_image(self.sup_sample['slice_2_img_omega'],
                                                                             self.sup_sample['slice_2_k_omega'],
                                                                             self.sup_sample['mask_omega'],
                                                                             self.sup_sample['slice_2_k'],
                                                                             torch.ones_like(
                                                                                 self.sup_sample['mask_omega'])
                                                                             )
        slice_2_img_rec_by_rec_model_2, slice_2_rec_loss_by_rec_model_2 = self.rec_model_2.forward_get_image(
            self.sup_sample['slice_2_img_omega'],
            self.sup_sample['slice_2_k_omega'],
            self.sup_sample['mask_omega'],
            self.sup_sample['slice_2_k'],
            torch.ones_like(self.sup_sample['mask_omega'])
            )
        sup_loss_rec = self.rec_sup_weight * (slice_2_rec_loss + slice_2_rec_loss_by_rec_model_2)

        return sup_loss_rec

    def sup_inter_forward(self):
        # sup interpolation
        slice_1 = torch.abs(real2complex_tensor(self.sup_sample['slice_1']))  # to get real-valued img
        slice_2 = torch.abs(real2complex_tensor(self.sup_sample['slice_2']))
        slice_3 = torch.abs(real2complex_tensor(self.sup_sample['slice_3']))
        slice_2_img_inter = self.inter_model.recon(slice_1, slice_3)
        inter_diff = slice_2_img_inter - slice_2
        sup_loss_inter = self.inter_sup_weight * self.imgspace_criterion(inter_diff, torch.zeros_like(inter_diff))

        return sup_loss_inter

    def unsup_rec_forward(self):
        # recon (we reocn all triplet slices to inter, but calculate loss only on the second one (slice 3))
        slice_1_img_rec, _ = self.rec_model.forward_get_image(self.unsup_sample['slice_1_img_omega'],
                                                    self.unsup_sample['slice_1_k_omega'],
                                                    self.unsup_sample['mask_omega'],
                                                    torch.zeros_like(self.unsup_sample['slice_1_k_omega']),
                                                    torch.ones_like(self.sup_sample['mask_omega'])
                        )
        slice_3_img_rec, _ = self.rec_model.forward_get_image(self.unsup_sample['slice_3_img_omega'],
                                                    self.unsup_sample['slice_3_k_omega'],
                                                    self.unsup_sample['mask_omega'],
                                                    torch.zeros_like(self.unsup_sample['slice_1_k_omega']),
                                                    torch.ones_like(self.sup_sample['mask_omega'])
                        )
        slice_5_img_rec, _ = self.rec_model.forward_get_image(self.unsup_sample['slice_5_img_omega'],
                                                    self.unsup_sample['slice_5_k_omega'],
                                                    self.unsup_sample['mask_omega'],
                                                    torch.zeros_like(self.unsup_sample['slice_1_k_omega']),
                                                    torch.ones_like(self.sup_sample['mask_omega'])
                        )
        slice_1_img_rec_2, _ = self.rec_model_2.forward_get_image(self.unsup_sample['slice_1_img_omega'],
                                                    self.unsup_sample['slice_1_k_omega'],
                                                    self.unsup_sample['mask_omega'],
                                                    torch.zeros_like(self.unsup_sample['slice_1_k_omega']),
                                                    torch.ones_like(self.sup_sample['mask_omega'])
                        )
        slice_3_img_rec_2, _ = self.rec_model_2.forward_get_image(self.unsup_sample['slice_3_img_omega'],
                                                    self.unsup_sample['slice_3_k_omega'],
                                                    self.unsup_sample['mask_omega'],
                                                    torch.zeros_like(self.unsup_sample['slice_1_k_omega']),
                                                    torch.ones_like(self.sup_sample['mask_omega'])
                        )
        slice_5_img_rec_2, _ = self.rec_model_2.forward_get_image(self.unsup_sample['slice_5_img_omega'],
                                                    self.unsup_sample['slice_5_k_omega'],
                                                    self.unsup_sample['mask_omega'],
                                                    torch.zeros_like(self.unsup_sample['slice_1_k_omega']),
                                                    torch.ones_like(self.sup_sample['mask_omega'])
                        )
        # dual-domain reconstruction consistency
        rec_loss = self.rec_unsup_weight * self.imgspace_criterion(slice_3_img_rec, slice_3_img_rec_2)

        # interpolation
        # to real, abs
        slice_1_img_rec = torch.abs(real2complex_tensor(slice_1_img_rec))
        slice_3_img_rec = torch.abs(real2complex_tensor(slice_3_img_rec))
        slice_5_img_rec = torch.abs(real2complex_tensor(slice_5_img_rec))
        # 1st order inter
        slice_2_img_inter = self.inter_model.recon(slice_1_img_rec, slice_3_img_rec)
        slice_4_img_inter = self.inter_model.recon(slice_3_img_rec, slice_5_img_rec)
        slice_inter = torch.cat([slice_2_img_inter, slice_4_img_inter], dim=0)

        # to real, abs
        slice_1_img_rec_2 = torch.abs(real2complex_tensor(slice_1_img_rec_2))
        slice_3_img_rec_2 = torch.abs(real2complex_tensor(slice_3_img_rec_2))
        slice_5_img_rec_2 = torch.abs(real2complex_tensor(slice_5_img_rec_2))
        # 1st order inter
        slice_2_img_inter_2 = self.inter_model.recon(slice_1_img_rec_2, slice_3_img_rec_2)
        slice_4_img_inter_2 = self.inter_model.recon(slice_3_img_rec_2, slice_5_img_rec_2)
        slice_inter_2 = torch.cat([slice_2_img_inter_2, slice_4_img_inter_2], dim=0)

        # dual-domain recon-interpolation consistency
        rec_inter_loss = self.rec_unsup_weight * self.imgspace_criterion(slice_inter, slice_inter_2)

        unsup_rec_loss = rec_loss + rec_inter_loss

        return unsup_rec_loss

    def unsup_inter_forward(self):
        # recon (we reocn all triplet slices to inter, but calculate loss only on the second one (slice 3))
        slice_1_img_rec, _ = self.rec_model.forward_get_image(self.unsup_sample['slice_1_img_omega'],
                                                              self.unsup_sample['slice_1_k_omega'],
                                                              self.unsup_sample['mask_omega'],
                                                              torch.zeros_like(self.unsup_sample['slice_1_k_omega']),
                                                              torch.ones_like(self.sup_sample['mask_omega'])
                                                              )
        slice_3_img_rec, _ = self.rec_model.forward_get_image(self.unsup_sample['slice_3_img_omega'],
                                                              self.unsup_sample['slice_3_k_omega'],
                                                              self.unsup_sample['mask_omega'],
                                                              torch.zeros_like(self.unsup_sample['slice_1_k_omega']),
                                                              torch.ones_like(self.sup_sample['mask_omega'])
                                                              )
        slice_5_img_rec, _ = self.rec_model.forward_get_image(self.unsup_sample['slice_5_img_omega'],
                                                              self.unsup_sample['slice_5_k_omega'],
                                                              self.unsup_sample['mask_omega'],
                                                              torch.zeros_like(self.unsup_sample['slice_1_k_omega']),
                                                              torch.ones_like(self.sup_sample['mask_omega'])
                                                              )

        # interpolation
        # to real, abs
        slice_1_img_rec = torch.abs(real2complex_tensor(slice_1_img_rec))
        slice_3_img_rec = torch.abs(real2complex_tensor(slice_3_img_rec))
        slice_5_img_rec = torch.abs(real2complex_tensor(slice_5_img_rec))
        # 1st order inter
        slice_2_img_inter = self.inter_model.recon(slice_1_img_rec, slice_3_img_rec)
        slice_4_img_inter = self.inter_model.recon(slice_3_img_rec, slice_5_img_rec)
        # 2nd order inter
        slice_3_img_inter = self.inter_model.recon(slice_2_img_inter, slice_4_img_inter)

        unsup_inter_loss = self.inter_unsup_weight * self.imgspace_criterion(slice_3_img_rec, slice_3_img_inter)

        return unsup_inter_loss

    def train_rec_forward(self):
        loss = 0.

        # for sup samples
        loss_sup = self.sup_rec_forward()

        # for unsup samples
        loss_unsup = self.unsup_rec_forward()

        loss = loss_sup + loss_unsup

        self.to_meters('sup_rec_loss', loss_sup.item())
        self.to_meters('unsup_rec_loss', loss_unsup.item())
        self.to_meters('rec_loss', loss.item())

        return loss
    def train_inter_forward(self):
        loss = 0.

        # for sup samples
        loss_sup = self.sup_inter_forward()

        # for unsup samples
        loss_unsup = self.unsup_inter_forward()

        loss = loss_sup + loss_unsup

        self.to_meters('sup_inter_loss', loss_sup.item())
        self.to_meters('unsup_inter_loss', loss_unsup.item())
        self.to_meters('inter_loss', loss.item())

        return loss

    def set_train_mode(self, mode):
        assert mode in ['rec', 'inter']

        self.train_mode = mode

        if self.train_mode == 'rec':
            for p in self.rec_model.parameters():
                p.requires_grad = True
            for p in self.rec_model_2.parameters():
                p.requires_grad = True
            for p in self.inter_model.parameters():
                p.requires_grad = False
        else:
            for p in self.rec_model.parameters():
                p.requires_grad = False
            for p in self.rec_model_2.parameters():
                p.requires_grad = False
            for p in self.inter_model.parameters():
                p.requires_grad = True

    def train_forward(self):
        assert self.train_mode in ['rec', 'inter']

        if self.train_mode == 'rec':
            return self.train_rec_forward()
        else:
            return self.train_inter_forward()

    def update(self):
        loss = self.train_forward()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in self.optimizers:
            optimizer.step()

    def inference(self):
        # recon (only recon the middle slice in the triplet slices)
        slice_2_img_rec, _ = self.rec_model.forward_get_image(self.inf_sample['slice_2_img_omega'],
                                                 self.inf_sample['slice_2_k_omega'],
                                                 self.inf_sample['mask_omega'],
                                                 self.inf_sample['slice_2_k'],
                                                 torch.ones_like(self.inf_sample['mask_omega'])
                                                 )
        slice_2_img_rec_2, _ = self.rec_model_2.forward_get_image(self.inf_sample['slice_2_img_omega'],
                                                 self.inf_sample['slice_2_k_omega'],
                                                 self.inf_sample['mask_omega'],
                                                 self.inf_sample['slice_2_k'],
                                                 torch.ones_like(self.inf_sample['mask_omega'])
                                                 )
        self.rec_gt = self.inf_sample['slice_2']
        self.rec_pred = slice_2_img_rec
        self.rec_pred_2 = slice_2_img_rec_2

        # interpolation (using gt slices as input)
        slice_1 = torch.abs(real2complex_tensor(self.inf_sample['slice_1']))  # to get real-valued img
        slice_2 = torch.abs(real2complex_tensor(self.inf_sample['slice_2']))
        slice_3 = torch.abs(real2complex_tensor(self.inf_sample['slice_3']))
        slice_2_img_inter = self.inter_model.recon(slice_1, slice_3)

        self.inter_gt = slice_2
        self.inter_pred = slice_2_img_inter

    def post_evaluation(self):
        # for rec slices
        # get magnitude images
        img_ref = torch.abs(torch.view_as_complex(self.rec_gt.permute(0, 2, 3, 1).contiguous()))
        img_output = torch.abs(torch.view_as_complex(self.rec_pred.permute(0, 2, 3, 1).contiguous()))
        img_output_2 = torch.abs(torch.view_as_complex(self.rec_pred_2.permute(0, 2, 3, 1).contiguous()))
        img_diff = img_output - img_ref
        img_diff_2 = img_output_2 - img_ref

        # calculate metrics
        psnr =  psnr_slice(img_ref, img_output)
        ssim = ssim_slice(img_ref, img_output)

        self.to_meters('val_rec_image_psnr', psnr)
        self.to_meters('val_rec_image_ssim', ssim)

        psnr_2 = psnr_slice(img_ref, img_output_2)
        ssim_2 = ssim_slice(img_ref, img_output_2)

        self.to_meters('val_rec_image_psnr_2', psnr_2)
        self.to_meters('val_rec_image_ssim_2', ssim_2)

        # for inter slices
        # get magnitude images
        img_inter_ref = self.inter_gt[:, 0]
        img_inter_output = self.inter_pred[:, 0]
        img_inter_diff = img_inter_output - img_inter_ref

        # calculate metrics
        psnr =  psnr_slice(img_inter_ref, img_inter_output)
        ssim = ssim_slice(img_inter_ref, img_inter_output)

        self.to_meters('val_inter_image_psnr', psnr)
        self.to_meters('val_inter_image_ssim', ssim)

        if self.save_test_vis:
            if not hasattr(self, 'cnt'):
                self.cnt = 0
            else:
                self.cnt += 1
            # rec
            plt.imsave(os.path.join(self.args.output_path, f'img_rec_ref_{self.cnt}.png'), img_ref.detach().cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_rec_output_{self.cnt}.png'), img_output.detach().cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_rec_diff_{self.cnt}.png'), img_diff.detach().cpu()[0], cmap='bwr', vmin=-0.3, vmax=0.3)
            plt.imsave(os.path.join(self.args.output_path, f'img_rec_diff_2_{self.cnt}.png'), img_diff_2.detach().cpu()[0], cmap='bwr', vmin=-0.3, vmax=0.3)
            # inter
            plt.imsave(os.path.join(self.args.output_path, f'img_inter_ref_{self.cnt}.png'), img_inter_ref.detach().cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_inter_output_{self.cnt}.png'), img_inter_output.detach().cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_inter_diff_{self.cnt}.png'), img_inter_diff.detach().cpu()[0], cmap='bwr', vmin=-0.3, vmax=0.3)


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
            for i, optimizer in enumerate(self.optimizers):
                log[f'lr_{i+1}'] = optimizer.param_groups[0]['lr']
        log.update(self.summarize_meters())

        return log


JRI = JointReconInterModel
