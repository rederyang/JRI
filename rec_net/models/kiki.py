import torch
from torch import nn
from torch.nn import functional as F

from rec_net.models.utils import DataConsistencyInKspace_I, DataConsistencyInKspace_K
from mri_tools import fft2_tensor, ifft2_tensor


def GenConvBlock(n_conv_layers, in_chan, out_chan, feature_maps):
    conv_block = [nn.Conv2d(in_chan, feature_maps, 3, 1, 1),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    for _ in range(n_conv_layers - 2):
        conv_block += [nn.Conv2d(feature_maps, feature_maps, 3, 1, 1),
                       nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    return nn.Sequential(*conv_block, nn.Conv2d(feature_maps, out_chan, 3, 1, 1))


class KIKI(nn.Module):
    def __init__(self, iters=2, k_blocks=25, i_blocks=25, in_ch=2, out_ch=2, fm=64):
        super(KIKI, self).__init__()

        conv_blocks_K = []
        conv_blocks_I = []

        for i in range(iters):
            conv_blocks_K.append(GenConvBlock(k_blocks, in_ch, out_ch, fm))
            conv_blocks_I.append(GenConvBlock(i_blocks, in_ch, out_ch, fm))

        self.conv_blocks_K = nn.ModuleList(conv_blocks_K)
        self.conv_blocks_I = nn.ModuleList(conv_blocks_I)
        self.n_iter = iters

        self.dc = DataConsistencyInKspace_I()

        self.criterion = nn.L1Loss()

    def forward(self, img_subset, k_subset, mask_subset, k_omega, mask_omega):
        K = k_subset
        I = img_subset

        for i in range(self.n_iter):
            K = self.conv_blocks_K[i](K)
            I = ifft2_tensor(K)
            I = I + self.conv_blocks_I[i](I)
            I, K = self.dc(I, k_subset, mask_subset)

        loss = self.criterion(K*mask_omega, k_omega)

        return I, loss
