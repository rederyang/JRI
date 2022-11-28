import os
import time

import torch
from torch import nn

import matplotlib.pyplot as plt

from mri_tools import *
from utils import *
from inter_net.RDN import RDN
from inter_net.Discriminator import Discriminator
from inter_net.base_network import BaseModel


class TSCNet(BaseModel):
    def build(self):
        self.feat_net_1 = RDN(c_in=self.args.c_in, c_out=self.args.c_feat, blocks=self.args.blocks,
                              layers=self.args.layers, grow_rate=self.args.grow_rate)
        self.feat_net_2 = RDN(c_in=self.args.c_in, c_out=self.args.c_feat, blocks=self.args.blocks,
                              layers=self.args.layers, grow_rate=self.args.grow_rate)
        self.recon_net = RDN(c_in=2 * self.args.c_feat, c_out=self.args.c_in, blocks=self.args.blocks,
                             layers=self.args.layers, grow_rate=self.args.grow_rate)

        self.optimizer = torch.optim.Adam(list(self.feat_net_1.parameters()) +
                                          list(self.feat_net_2.parameters()) +
                                          list(self.recon_net.parameters()),
                                          lr=self.args.lr)

        self.criterion = nn.MSELoss()

    def recon(self, slice_1, slice_2):
        feat_1 = self.feat_net_1.forward(slice_1)
        feat_2 = self.feat_net_2.forward(slice_2)
        feat = torch.cat([feat_1, feat_2], dim=-3)
        middle_slice = self.recon_net.forward(feat)

        return middle_slice

    def set_input(self, mode, data_batch):
        """
        all shape == [bs, 2, x, y]
        """
        if isinstance(data_batch, tuple):
            data_batch = list(data_batch)
            for i in range(len(data_batch)):
                data_batch[i] = [item.to(self.rank, non_blocking=True) for item in data_batch[i]]
        else:
            data_batch = [item.to(self.rank, non_blocking=True) for item in data_batch]

        if mode == 'pre':
            self.input_slice_1, self.input_slice_2, self.target_slice = data_batch
        elif mode == 'self':
            self.input_slice_1, self.input_slice_2, self.target_slice = data_batch[0]
            self.cyc_slice_1, self.cyc_slice_2, self.cyc_slice_3 = data_batch[1]
        elif mode == 'test' or mode == 'val':
            self.input_slice_1, self.input_slice_2, self.target_slice = data_batch
        else:
            assert False

    def train_forward_backward(self, mode):
        loss = 0.

        if mode == 'pre':
            self.optimizer.zero_grad()  # 1
            output_slice = self.recon(self.input_slice_1, self.input_slice_2)
            diff = output_slice - self.target_slice
            loss = self.criterion(diff, torch.zeros_like(diff))
            loss.backward()  # 2
            self.optimizer.step()  # 3
        elif mode == 'self':
            self.optimizer.zero_grad()  # 1
            # loss mse
            output_slice = self.recon(self.input_slice_1, self.input_slice_2)
            diff = output_slice - self.target_slice
            loss_mse = self.criterion(diff, torch.zeros_like(diff))
            loss_mse.backward()  # 2
            # loss cycle consistency
            output_slice_12 = self.recon(self.cyc_slice_1, self.cyc_slice_2)
            output_slice_23 = self.recon(self.cyc_slice_2, self.cyc_slice_3)
            output_slice_2 = self.recon(output_slice_12, output_slice_23)
            diff = output_slice_2 - self.cyc_slice_2
            loss_cyc = self.criterion(diff, torch.zeros_like(diff))
            loss_cyc.backward()  # 2
            self.optimizer.step()  # 3
            loss = loss_mse + loss_cyc

        return loss

    def update(self, mode):
        loss = self.train_forward_backward(mode)

        return loss

    def inference(self):
        output_slice = self.recon(self.input_slice_1, self.input_slice_2)
        diff = output_slice - self.target_slice
        loss = self.criterion(diff, torch.zeros_like(diff))

        return output_slice, loss

    def post_evaluation(self):
        # get magnitude images
        # gt_slice = torch.abs(torch.view_as_complex(self.target_slice.permute(0, 2, 3, 1).contiguous()))
        # output_slice = torch.abs(torch.view_as_complex(self.output_slice.permute(0, 2, 3, 1).contiguous()))

        output_slice = self.output_slice[:, 0]
        gt_slice = self.target_slice[:, 0]

        # plt.figure()
        # plt.imsave(f'output_slice.png', output_slice[0], cmap='gray')
        # plt.show()
        #
        # plt.figure()
        # plt.imsave(f'gt_slice.png', gt_slice[0], cmap='gray')
        # plt.show()

        diff = output_slice - gt_slice

        # print(output_slice.shape)
        # print(gt_slice.shape)

        # calculate metrics
        psnr = psnr_slice(gt_slice, output_slice)
        ssim = ssim_slice(gt_slice, output_slice)

        # print(psnr)
        # print(ssim)

        if self.save_test_vis:
            if not hasattr(self, 'cnt'):
                self.cnt = 0
            else:
                self.cnt += 1
            plt.imsave(os.path.join(self.args.output_path, f'img_gt_{self.cnt}.png'), gt_slice.cpu()[0], cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_output_{self.cnt}.png'), output_slice.cpu()[0],
                       cmap='gray')
            plt.imsave(os.path.join(self.args.output_path, f'img_diff_{self.cnt}.png'), diff.cpu()[0], cmap='bwr',
                       vmin=-0.3, vmax=0.3)

        return psnr, ssim

    def run_one_epoch(self, mode, dataloader):
        assert mode in ['pre', 'self', 'val', 'test']

        tik = time.time()

        loss, psnr, ssim = 0., 0., 0.

        for iter_num, data_batch in enumerate(dataloader):

            self.set_input(mode, data_batch)

            if mode == 'pre' or mode == 'self':
                batch_loss = self.update(mode)
            else:
                self.output_slice, batch_loss = self.inference()
                _psnr, _ssim = self.post_evaluation()
                psnr += _psnr
                ssim += _ssim

            loss += batch_loss.item()

        loss /= len(dataloader)

        log = dict()
        log['epoch'] = self.epoch
        log['loss'] = loss
        if mode not in ['val', 'test']:
            log['lr'] = self.optimizer.param_groups[0]['lr']
        else:
            psnr /= len(dataloader)
            ssim /= len(dataloader)
            log['psnr'] = psnr
            log['ssim'] = ssim

        tok = time.time()

        log['time'] = tok - tik

        return log


# noinspection PyCallingNonCallable
class AdversarialTSCNet(TSCNet):
    def build(self):
        self.feat_net_1 = RDN(c_in=self.args.c_in, c_out=self.args.c_feat, blocks=self.args.blocks,
                              layers=self.args.layers, grow_rate=self.args.grow_rate)
        self.feat_net_2 = RDN(c_in=self.args.c_in, c_out=self.args.c_feat, blocks=self.args.blocks,
                              layers=self.args.layers, grow_rate=self.args.grow_rate)
        self.recon_net = RDN(c_in=2 * self.args.c_feat, c_out=self.args.c_in, blocks=self.args.blocks,
                             layers=self.args.layers, grow_rate=self.args.grow_rate)

        self.optimizer = torch.optim.Adam(list(self.feat_net_1.parameters()) +
                                          list(self.feat_net_2.parameters()) +
                                          list(self.recon_net.parameters()),
                                          lr=self.args.lr)

        self.criterion = nn.MSELoss()

        self.d_net = Discriminator()

        self.optimizer_d = torch.optim.Adam(list(self.d_net.parameters()), lr=self.args.lr)

        self.criterion_adversarial = nn.BCEWithLogitsLoss()

    def train_forward_backward(self, mode):
        loss = 0.

        if mode == 'pre':
            self.optimizer.zero_grad()  # 1
            output_slice = self.recon(self.input_slice_1, self.input_slice_2)
            diff = output_slice - self.target_slice
            loss = self.criterion(diff, torch.zeros_like(diff))
            loss.backward()  # 2
            self.optimizer.step()  # 3
        elif mode == 'self':
            self.optimizer.zero_grad()  # 1
            # 1. mse loss
            output_slice = self.recon(self.input_slice_1, self.input_slice_2)
            diff = output_slice - self.target_slice
            loss_mse = self.criterion(diff, torch.zeros_like(diff))
            loss_mse.backward()  # 2

            # 2. cyc loss
            # generator forward
            output_slice_12 = self.recon(self.cyc_slice_1, self.cyc_slice_2)
            output_slice_23 = self.recon(self.cyc_slice_2, self.cyc_slice_3)
            output_slice_2 = self.recon(output_slice_12, output_slice_23)

            ## discriminator training
            # construct labels
            batch_size = output_slice_2.shape[0]
            real_label = torch.full([batch_size,], 1.0, dtype=output_slice_2.dtype, device=output_slice_2.device)
            fake_label = torch.full([batch_size,], 0.0, dtype=output_slice_2.dtype, device=output_slice_2.device)
            # activate net_d gradient
            for d_parameters in self.d_net.parameters():
                d_parameters.requires_grad = True
            # real sample loss and backward
            real_output = self.d_net(self.cyc_slice_2)
            d_loss_real = self.criterion_adversarial(real_output, real_label)
            # fake sample loss and backward
            fake_output = self.d_net(output_slice_2.detach().clone())
            d_loss_fake = self.criterion_adversarial(fake_output, fake_label)
            # optimize
            self.d_net.zero_grad(set_to_none=True)  # _1
            d_loss_real.backward(retain_graph=True)  # _2
            d_loss_fake.backward()  # _2
            self.optimizer_d.step()  # _3
            # deactivate net_d gradient
            for d_parameters in self.d_net.parameters():
                d_parameters.requires_grad = False

            ## generator training
            # cycle consistency loss
            diff = output_slice_2 - self.cyc_slice_2
            loss_cyc_mse = self.criterion(diff, torch.zeros_like(diff))
            loss_cyc_mse.backward(retain_graph=True)  # 2
            # adversarial loss
            loss_cyc_adv = self.args.adv_weight * self.criterion_adversarial(self.d_net(output_slice_2), real_label)
            loss_cyc_adv.backward()  # 2

            self.optimizer.step()  # 3

            loss = loss_mse + loss_cyc_mse + loss_cyc_adv

        return loss

    def load(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(self.rank))
        self.epoch = checkpoint['epoch']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'optimizer_d' in checkpoint.keys():
            self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.best_target_metric = checkpoint['best_metric']
        self.load_state_dict(checkpoint['model'], strict=False)  # might be missing keys for discriminator

    def save(self, model_path=None):
        # save checkpoint
        checkpoint = {
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'best_metric': self.best_target_metric,
            'model': self.state_dict()
        }
        if model_path is None:
            model_path = self.model_path
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(checkpoint, model_path)