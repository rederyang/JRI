import os
import time

import torch
from torch import nn

import matplotlib.pyplot as plt

from mri_tools import *
from utils import *
from rec_net.models import *
from rec_net.base_network import BaseModel


class SemisupervisedParallelKINetworkV3(BaseModel):
    def build(self):
        self.network_k = du_recurrent_model.KRNet(self.args)
        self.network_k.initialize()
        self.network_i = du_recurrent_model.IRNet(self.args)
        self.network_i.initialize()

        # self.network_k = unet.KUnet()
        # self.network_i = unet.IUnet()

        self.optimizer = torch.optim.Adam(list(self.network_k.parameters()) +
                                          list(self.network_i.parameters()),
                                          lr=self.args.lr)

        self.criterion = nn.MSELoss()

    def attach_meters(self):

        self.unsup_train_network_k_loss_meter = AverageMeter()
        self.unsup_train_network_i_loss_meter = AverageMeter()
        self.unsup_train_aux_loss_meter = AverageMeter()
        self.unsup_train_loss_meter = AverageMeter()

        self.sup_train_network_k_loss_meter = AverageMeter()
        self.sup_train_network_i_loss_meter = AverageMeter()
        self.sup_train_loss_meter = AverageMeter()

        self.train_loss_meter = AverageMeter()

        # self.train_image_1_psnr_meter = AverageMeter()
        # self.train_image_2_psnr_meter = AverageMeter()
        # self.train_image_1_ssim_meter = AverageMeter()
        # self.train_image_2_ssim_meter = AverageMeter()

        self.val_network_k_loss_meter = AverageMeter()
        self.val_network_i_loss_meter = AverageMeter()
        # self.val_aux_loss_meter = AverageMeter()

        self.val_loss_meter = AverageMeter()

        self.val_image_1_psnr_meter = AverageMeter()
        self.val_image_2_psnr_meter = AverageMeter()
        self.val_image_1_ssim_meter = AverageMeter()
        self.val_image_2_ssim_meter = AverageMeter()

    def set_input(self, mode, data_batch):
        """
        all shape == [bs, 2, x, y]
        """
        img_full = data_batch[0].to(self.rank, non_blocking=True)  # full sampled image [bs, 1, x, y]
        img_full = torch.view_as_real(img_full[:, 0]).permute(0, 3, 1, 2).contiguous()  # -> [bs, 2, x, y]
        mask_omega = data_batch[1].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        mask_subset1 = data_batch[2].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()
        mask_subset2 = data_batch[3].to(self.rank, non_blocking=True).permute(0, 3, 1, 2).contiguous()

        if mode == 'train':  # save unsup and sup sample idx, because they have different behaviours when forward
            _is_unsupervised = data_batch[4]['_is_unsupervised']  # [bs, ]
            unsupervised_idx = _is_unsupervised.nonzero()[:, 0]
            supervised_idx = torch.logical_not(_is_unsupervised).nonzero()[:, 0]
            self.supervised_idx = supervised_idx
            self.unsupervised_idx = unsupervised_idx

        self.img_full = img_full
        self.k_full = fft2_tensor(img_full)

        self.k_omega = self.k_full * mask_omega
        self.img_omega = ifft2_tensor(self.k_omega)
        self.mask_omega = mask_omega

        self.k_subset1 = self.k_omega * mask_subset1
        self.img_subset1 = ifft2_tensor(self.k_subset1)
        self.mask_subset1 = mask_subset1

        self.k_subset2 = self.k_omega * mask_subset2
        self.img_subset2 = ifft2_tensor(self.k_subset2)
        self.mask_subset2 = mask_subset2

    def sup_train_forward(self):
        output_k, loss_k_branch = self.network_k.forward(
            self.img_omega[self.supervised_idx],
            self.k_omega[self.supervised_idx],
            self.mask_omega[self.supervised_idx],
            self.k_full[self.supervised_idx],
            torch.ones_like(self.mask_omega[self.supervised_idx])
        )
        output_i, loss_i_branch = self.network_i.forward(
            self.img_omega[self.supervised_idx],
            self.k_omega[self.supervised_idx],
            self.mask_omega[self.supervised_idx],
            self.k_full[self.supervised_idx],
            torch.ones_like(self.mask_omega[self.supervised_idx])
        )

        loss_sup = loss_k_branch + loss_i_branch

        self.sup_train_network_k_loss_meter.update(loss_k_branch.item())
        self.sup_train_network_i_loss_meter.update(loss_i_branch.item())
        self.sup_train_loss_meter.update(loss_sup.item())

        return loss_sup

    def unsup_train_forward(self):
        output_k, loss_k_branch = self.network_k.forward(
            self.img_subset1[self.unsupervised_idx],
            self.k_subset1[self.unsupervised_idx],
            self.mask_subset1[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )
        output_i, loss_i_branch = self.network_i.forward(
            self.img_subset2[self.unsupervised_idx],
            self.k_subset2[self.unsupervised_idx],
            self.mask_subset2[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )

        # some loss term based on the above (like diff loss)
        diff = (output_k - fft2_tensor(output_i)) * (1 - self.mask_omega[self.unsupervised_idx])
        diff_loss = 0.01 * self.criterion(diff, torch.zeros_like(diff))
        loss_unsup = loss_k_branch + loss_i_branch + diff_loss

        self.unsup_train_network_k_loss_meter.update(loss_k_branch.item())
        self.unsup_train_network_i_loss_meter.update(loss_i_branch.item())
        self.unsup_train_aux_loss_meter.update(diff_loss.item())
        self.unsup_train_loss_meter.update(loss_unsup.item())

        return loss_unsup

    def train_forward(self):
        loss_sup, loss_unsup = 0., 0.

        # for sup samples
        if len(self.supervised_idx) > 0:
            loss_sup = self.sup_train_forward()

        # for unsup samples
        if len(self.unsupervised_idx) > 0:
            loss_unsup = self.unsup_train_forward()

        loss = loss_sup + loss_unsup

        self.train_loss_meter.update(loss.item())

        return loss

    def update(self):
        loss = self.train_forward()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def inference(self):
        output_k, loss_k_branch = self.network_k.forward(
            self.img_omega,
            self.k_omega,
            self.mask_omega,
            self.k_full,
            torch.ones_like(self.k_full)
        )
        output_i, loss_i_branch = self.network_i.forward(
            self.img_omega,
            self.k_omega,
            self.mask_omega,
            self.k_full,
            torch.ones_like(self.k_full)
        )

        loss = loss_k_branch + loss_i_branch

        self.val_network_k_loss_meter.update(loss_k_branch.item())
        self.val_network_i_loss_meter.update(loss_i_branch.item())
        self.val_loss_meter.update(loss.item())

        self.output_i_1 = ifft2_tensor(output_k)
        self.output_i_2 = output_i

    def post_evaluation(self):
        # get magnitude images
        img_full = torch.abs(torch.view_as_complex(self.img_full.permute(0, 2, 3, 1).contiguous()))
        img_omega = torch.abs(torch.view_as_complex(self.img_omega.permute(0, 2, 3, 1).contiguous()))
        output_i_1 = torch.abs(torch.view_as_complex(self.output_i_1.permute(0, 2, 3, 1).contiguous()))
        output_i_2 = torch.abs(torch.view_as_complex(self.output_i_2.permute(0, 2, 3, 1).contiguous()))

        img_diff_2 = output_i_2 - img_full

        # calculate metrics
        psnr_1 = psnr_slice(img_full, output_i_1)
        ssim_1 = ssim_slice(img_full, output_i_1)
        psnr_2 = psnr_slice(img_full, output_i_2)
        ssim_2 = ssim_slice(img_full, output_i_2)

        self.val_image_1_psnr_meter.update(psnr_1)
        self.val_image_2_psnr_meter.update(psnr_2)
        self.val_image_1_ssim_meter.update(ssim_1)
        self.val_image_2_ssim_meter.update(ssim_2)

        if self.save_test_vis:
            if not hasattr(self, 'cnt'):
                self.cnt = 0
            else:
                self.cnt += 1
            plt.imsave(os.path.join(self.args.output_path, f'img_full_{self.cnt}.png'), img_full.cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_omega_{self.cnt}.png'), img_omega.cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_output1_{self.cnt}.png'), output_i_1.cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_output2_{self.cnt}.png'), output_i_2.cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_diff2_{self.cnt}.png'), img_diff_2.cpu()[0], cmap='bwr', vmin=-0.3, vmax=0.3)

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
            log['lr'] = self.optimizer.param_groups[0]['lr']
        log.update(self.summarize_meters())

        return log


class SemisupervisedParallelKINetworkV3Unet(SemisupervisedParallelKINetworkV3):
    def build(self):
        self.network_k = unet.KUnet()
        self.network_i = unet.IUnet()

        self.optimizer = torch.optim.Adam(list(self.network_k.parameters()) +
                                          list(self.network_i.parameters()),
                                          lr=self.args.lr)

        self.criterion = nn.MSELoss()


class SemisupervisedParallelKINetworkV4(SemisupervisedParallelKINetworkV3):
    """
    Compared to V3,
    (1) Use share-weight model, so one single model totally. But this model has a K-subnet and a I-subnet
    (2) Dual-domain consistency by: K-K-K-K -> I-K-I-K <- I-I-I-I
    (3) Because of (2), dual-input consistency and dual-domain consistency are decoupled
    """
    def build(self):
        hybrid_model = du_recurrent_model.HybridRecurrentModel(self.args)
        hybrid_model.initialize()
        self.network_up = hybrid_model
        self.network_down = hybrid_model

        self.optimizer = torch.optim.Adam(hybrid_model.parameters(),
                                          lr=self.args.lr)

        self.criterion = nn.L1Loss()  # TODO: here we use L1loss

    def sup_train_forward(self):
        """train_forward for sup samples"""
        output_sup, loss_sup = self.network_down.dual_space_forward(
            self.img_omega[self.supervised_idx],
            self.k_omega[self.supervised_idx],
            self.mask_omega[self.supervised_idx],
            self.k_full[self.supervised_idx],
            torch.ones_like(self.k_full[self.supervised_idx])
        )

        return loss_sup

    def unsup_train_forward(self):
        """train_forward for unsup samples"""
        # for unsup samples
        output_up, loss_omega_up, loss_ddc_up = self.network_up.hybrid_forward(
            self.img_subset1[self.unsupervised_idx],
            self.k_subset1[self.unsupervised_idx],
            self.mask_subset1[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )
        output_down, loss_omega_down, loss_ddc_down = self.network_down.hybrid_forward(
            self.img_subset2[self.unsupervised_idx],
            self.k_subset2[self.unsupervised_idx],
            self.mask_subset2[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )

        # omega loss
        loss_omega = loss_omega_up + loss_omega_down

        # dual-domain consistency loss
        loss_ddc = loss_ddc_up + loss_ddc_down

        # dual-input loss, outside mask_omega
        diff = (fft2_tensor(output_up) - fft2_tensor(output_down)) * (1 - self.mask_omega[self.unsupervised_idx])
        loss_di = self.criterion(diff, torch.zeros_like(diff))

        loss_unsup = loss_omega + 0.01 * loss_ddc + 0.01 * loss_di

        return loss_unsup

    def inference(self):
        output_i, loss_omega = self.network_down.dual_space_forward(
            self.img_omega,
            self.k_omega,
            self.mask_omega,
            self.k_full,
            torch.ones_like(self.mask_omega)
        )

        output_i_1 = output_i_2 = output_i
        loss = loss_omega

        return output_i_1, output_i_2, loss


class SemisupervisedParallelKINetworkV5(SemisupervisedParallelKINetworkV3):
    """
    Compared to V4,
    (1) Remove dual-domain consistency, thus no hybrid-forward conducted.
    """
    def build(self):
        hybrid_model = du_recurrent_model.HybridRecurrentModel(self.args)
        hybrid_model.initialize()
        self.network_up = hybrid_model
        self.network_down = hybrid_model

        self.optimizer = torch.optim.Adam(hybrid_model.parameters(),
                                          lr=self.args.lr)

        self.criterion = nn.L1Loss()  # TODO: here we use L1loss

    def sup_train_forward(self):
        """train_forward for sup samples"""
        output_sup, loss_sup = self.network_down.dual_space_forward(
            self.img_omega[self.supervised_idx],
            self.k_omega[self.supervised_idx],
            self.mask_omega[self.supervised_idx],
            self.k_full[self.supervised_idx],
            torch.ones_like(self.k_full[self.supervised_idx])
        )

        return loss_sup

    def unsup_train_forward(self):
        """train_forward for unsup samples"""
        # for unsup samples
        output_up, loss_omega_up = self.network_up.dual_space_forward(
            self.img_subset1[self.unsupervised_idx],
            self.k_subset1[self.unsupervised_idx],
            self.mask_subset1[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )
        output_down, loss_omega_down = self.network_down.dual_space_forward(
            self.img_subset2[self.unsupervised_idx],
            self.k_subset2[self.unsupervised_idx],
            self.mask_subset2[self.unsupervised_idx],
            self.k_omega[self.unsupervised_idx],
            self.mask_omega[self.unsupervised_idx]
        )

        # omega loss
        loss_omega = loss_omega_up + loss_omega_down

        # dual-input loss, outside mask_omega
        diff = (fft2_tensor(output_up) - fft2_tensor(output_down)) * (1 - self.mask_omega[self.unsupervised_idx])
        loss_di = self.criterion(diff, torch.zeros_like(diff))

        loss_unsup = loss_omega + 0.01 * loss_di

        return loss_unsup

    def inference(self):
        output_i, loss_omega = self.network_down.dual_space_forward(
            self.img_omega,
            self.k_omega,
            self.mask_omega,
            self.k_full,
            torch.ones_like(self.mask_omega)
        )

        output_i_1 = output_i_2 = output_i
        loss = loss_omega

        return output_i_1, output_i_2, loss


class SemisupervisedKNetworkV3(SemisupervisedParallelKINetworkV3):
    def build(self):
        self.network_k = du_recurrent_model.KRNet(self.args)
        self.network_k.initialize()

        self.optimizer = torch.optim.Adam(list(self.network_k.parameters()),
                                          lr=self.args.lr)

        self.criterion = nn.MSELoss()

    def sup_train_forward(self):
        output_k, loss_k_branch = self.network_k.forward(
            self.img_omega[self.supervised_idx],
            self.k_omega[self.supervised_idx],
            self.mask_omega[self.supervised_idx],
            self.k_full[self.supervised_idx],
            torch.ones_like(self.mask_omega[self.supervised_idx])
        )

        loss_sup = loss_k_branch

        self.sup_train_network_k_loss_meter.update(loss_k_branch.item())
        self.sup_train_loss_meter.update(loss_sup.item())

        return loss_sup

    def unsup_train_forward(self):
        raise NotImplementedError

    def inference(self):
        output_k, loss_k_branch = self.network_k.forward(
            self.img_omega,
            self.k_omega,
            self.mask_omega,
            self.k_full,
            torch.ones_like(self.k_full)
        )

        loss = loss_k_branch

        self.val_network_k_loss_meter.update(loss_k_branch.item())
        self.val_loss_meter.update(loss.item())

        self.output_i_1 = ifft2_tensor(output_k)
        self.output_i_2 = ifft2_tensor(output_k)


class SemisupervisedKNetworkV4(SemisupervisedKNetworkV3):
    def build(self):
        self.network_k = du_recurrent_model.RecurrentOrderModel(self.args, k_last=True)

        self.optimizer = torch.optim.Adam(list(self.network_k.parameters()),
                                          lr=self.args.lr)


class SemisupervisedINetworkV3(SemisupervisedParallelKINetworkV3):
    def build(self):
        self.network_i = du_recurrent_model.IRNet(self.args)
        self.network_i.initialize()

        self.optimizer = torch.optim.Adam(list(self.network_i.parameters()),
                                          lr=self.args.lr)

        self.criterion = nn.MSELoss()

    def sup_train_forward(self):
        output_i, loss_i_branch = self.network_i.forward(
            self.img_omega[self.supervised_idx],
            self.k_omega[self.supervised_idx],
            self.mask_omega[self.supervised_idx],
            self.k_full[self.supervised_idx],
            torch.ones_like(self.mask_omega[self.supervised_idx])
        )

        loss_sup = loss_i_branch

        self.sup_train_network_i_loss_meter.update(loss_i_branch.item())
        self.sup_train_loss_meter.update(loss_sup.item())

        return loss_sup

    def unsup_train_forward(self):
        raise NotImplementedError

    def inference(self):
        output_i, loss_i_branch = self.network_i.forward(
            self.img_omega,
            self.k_omega,
            self.mask_omega,
            self.k_full,
            torch.ones_like(self.k_full)
        )

        loss = loss_i_branch

        self.val_network_i_loss_meter.update(loss_i_branch.item())
        self.val_loss_meter.update(loss.item())

        self.output_i_1 = output_i
        self.output_i_2 = output_i


class SemisupervisedINetworkV4(SemisupervisedINetworkV3):
    def build(self):
        self.network_i = du_recurrent_model.RecurrentOrderModel(self.args)

        self.optimizer = torch.optim.Adam(list(self.network_i.parameters()),
                                          lr=self.args.lr)


V3 = SemisupervisedParallelKINetworkV3
V3U = SemisupervisedParallelKINetworkV3Unet
KV3 = SemisupervisedKNetworkV3
KV4 = SemisupervisedKNetworkV4
IV3 = SemisupervisedINetworkV3
IV4 = SemisupervisedINetworkV4
