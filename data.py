#!/usr/bin/env python3

import os

import nibabel as nib
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import scipy.io as sio

from mri_tools import *


def center_crop(data, shape):
    if shape[0] <= data.shape[-2]:
        w_from = (data.shape[-2] - shape[0]) // 2
        w_to = w_from + shape[0]
        data = data[..., w_from:w_to, :]
    else:
        w_before = (shape[0] - data.shape[-2]) // 2
        w_after = shape[0] - data.shape[-2] - w_before
        pad = [(0, 0)] * data.ndim
        pad[-2] = (w_before, w_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    if shape[1] <= data.shape[-1]:
        h_from = (data.shape[-1] - shape[1]) // 2
        h_to = h_from + shape[1]
        data = data[..., :, h_from:h_to]
    else:
        h_before = (shape[1] - data.shape[-1]) // 2
        h_after = shape[1] - data.shape[-1] - h_before
        pad = [(0, 0)] * data.ndim
        pad[-1] = (h_before, h_after)
        data = np.pad(data, pad_width=pad, mode='constant', constant_values=0)
    return data


def crop_np(data: np.array, shape: [tuple, np.array]) -> np.array:
    if isinstance(shape, tuple):
        shape = np.array(shape)
    start = (data.shape[-3:] - shape) // 2
    end = start + shape
    data = data[..., start[-3]:end[-3], start[-2]:end[-2], start[-1]:end[-1]]

    return data


def pad_np(data, shape):
    h, w, d = data.shape
    _h, _w, _d = shape
    pad = (
        ((_h - h) // 2, _h - h - (_h - h) // 2),
        ((_w - w) // 2, _w - w - (_w - w) // 2),
        ((_d - d) // 2, _d - d - (_d - d) // 2),
    )
    data = np.pad(data, pad_width=pad, mode='constant')
    return data


def pad_torch(data, shape):
    h, w, d = data.shape
    _h, _w, _d = shape
    pad = (
        (_h - h) // 2, _h - h - (_h - h) // 2,
        (_w - w) // 2, _w - w - (_w - w) // 2,
        (_d - d) // 2, _d - d - (_d - d) // 2,
    )
    data = torch.nn.functional.pad(data, pad=pad[::-1], mode='constant')  # pad order is reversed!
    return data


def downsample(volume, dim):  # FIXME: this can be wrong! Maybe we should do this in k-space
    # 1st dim is channel dim
    if dim == 1:
        return volume[:, ::2]
    elif dim == 2:
        return volume[:, :, ::2]
    elif dim == 3:
        return volume[:, :, :, ::2]


class VolumeDataset(torch.utils.data.Dataset):
    def __init__(self, volume, crop=None, pad=None, q=0):
        super().__init__()
        assert q < 0.5
        self.volume = volume
        self.crop = crop
        self.q = q

        nii = nib.load(volume)
        array = nib.as_closest_canonical(nii).get_fdata()  # convert to RAS

        array = array.astype(np.float32)  # dtype

        array = array / np.max(array)  # normalization

        if crop is not None:
            array = crop_np(array, crop)

        if pad is not None:
            array = pad_np(array, pad)

        # array = array.astype(np.complex64)

        # # add num on each slice in each direction, to check
        # from PIL import Image
        # from PIL import ImageDraw
        # for i in range(array.shape[-1]):
        #     img = Image.fromarray(array[..., i])
        #     I1 = ImageDraw.Draw(img)
        #     I1.text((28, 36), f"{i}", fill=1.0)
        #     slice_array = np.array(img)
        #     array[..., i] = slice_array
        #
        #     img = Image.fromarray(array[..., i, :])
        #     I1 = ImageDraw.Draw(img)
        #     I1.text((28, 36), f"{i}", fill=1.0)
        #     slice_array = np.array(img)
        #     array[..., i, :] = slice_array
        #
        #     img = Image.fromarray(array[..., i, :, :])
        #     I1 = ImageDraw.Draw(img)
        #     I1.text((28, 36), f"{i}", fill=1.0)
        #     slice_array = np.array(img)
        #     array[..., i, :, :] = slice_array

        torch_array_padded = torch.from_numpy(array)
        self.array = torch_array_padded[None, ...]  # add channel dim

        length = self.array.shape[3]
        self.start = round(length * self.q)
        self.end = length - self.start


class ReconVolumeDataset(VolumeDataset):
    def __init__(self, mask_omega_path, mask_subset_1_path, mask_subset_2_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.array = self.array.type(torch.complex64)

        self.mask_under = np.array(sio.loadmat(mask_omega_path)['mask'])
        self.s_mask_up = np.array(sio.loadmat(mask_subset_1_path)['mask'])
        self.s_mask_down = np.array(sio.loadmat(mask_subset_2_path)['mask'])

        self.mask_net_up = self.mask_under * self.s_mask_up
        self.mask_net_down = self.mask_under * self.s_mask_down

        self.mask_under = np.stack((self.mask_under, self.mask_under), axis=-1)
        self.mask_omega = torch.from_numpy(self.mask_under).float()
        self.mask_net_up = np.stack((self.mask_net_up, self.mask_net_up), axis=-1)
        self.mask_subset_1 = torch.from_numpy(self.mask_net_up).float()
        self.mask_net_down = np.stack((self.mask_net_down, self.mask_net_down), axis=-1)
        self.mask_subset_2 = torch.from_numpy(self.mask_net_down).float()

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        slice = torch.index_select(self.array, 3,
                                           torch.tensor([index + self.start])).squeeze(dim=3)

        return slice, self.mask_omega, self.mask_subset_1, self.mask_subset_2, {'_is_unsupervised': True}


class ThickVolumeDataset(VolumeDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thick_array = downsample(self.array, dim=3)  # to simulate thick slice cases
        length_after_downsample = self.thick_array.shape[3]
        self.start = round(length_after_downsample * self.q)
        self.end = length_after_downsample - self.start


class ReconThickVolumeDataset(ThickVolumeDataset):
    def __init__(self, mask_omega_path, mask_subset_1_path, mask_subset_2_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thick_array = self.thick_array.type(torch.complex64)

        self.mask_under = np.array(sio.loadmat(mask_omega_path)['mask'])
        self.s_mask_up = np.array(sio.loadmat(mask_subset_1_path)['mask'])
        self.s_mask_down = np.array(sio.loadmat(mask_subset_2_path)['mask'])

        self.mask_net_up = self.mask_under * self.s_mask_up
        self.mask_net_down = self.mask_under * self.s_mask_down

        self.mask_under = np.stack((self.mask_under, self.mask_under), axis=-1)
        self.mask_omega = torch.from_numpy(self.mask_under).float()
        self.mask_net_up = np.stack((self.mask_net_up, self.mask_net_up), axis=-1)
        self.mask_subset_1 = torch.from_numpy(self.mask_net_up).float()
        self.mask_net_down = np.stack((self.mask_net_down, self.mask_net_down), axis=-1)
        self.mask_subset_2 = torch.from_numpy(self.mask_net_down).float()

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, index):
        slice = torch.index_select(self.thick_array, 3,
                                           torch.tensor([index + self.start])).squeeze(dim=3)

        return slice, self.mask_omega, self.mask_subset_1, self.mask_subset_2, {'_is_unsupervised': True}


class InterPretrainThickVolumeDataset(ThickVolumeDataset):
    def __init__(self, dim, *args, **kwargs):
        """
        down-sample along s or c axe, to construct interpolation pairs
        :param dim: along which dimension to down-sample
        """
        assert dim != 0
        super().__init__(*args, **kwargs)
        self.lr_thick_array = downsample(self.thick_array, dim=dim)
        self.dim = dim

    def __len__(self):
        return self.end - self.start - 1

    def __getitem__(self, index):
        input_slice_1 = torch.index_select(self.lr_thick_array, self.dim,
                                           torch.tensor([index + self.start])).squeeze(dim=self.dim)
        input_slice_2 = torch.index_select(self.lr_thick_array, self.dim,
                                           torch.tensor([index + self.start + 1])).squeeze(dim=self.dim)
        target_slice = torch.index_select(self.thick_array, self.dim,
                                          torch.tensor([(index + self.start) * 2 + 1])).squeeze(dim=self.dim)

        return input_slice_1, input_slice_2, target_slice


class InterInferenceThickVolumeDataset(ThickVolumeDataset):
    def __len__(self):
        return self.end - self.start - 1

    def __getitem__(self, index):
        input_slice_1 = torch.index_select(self.thick_array, 3,
                                           torch.tensor([index + self.start])).squeeze(dim=3)
        input_slice_2 = torch.index_select(self.thick_array, 3,
                                           torch.tensor([index + self.start + 1])).squeeze(dim=3)
        target_slice = torch.index_select(self.array, 3,
                                          torch.tensor([(index + self.start) * 2 + 1])).squeeze(dim=3)

        return input_slice_1, input_slice_2, target_slice


class JointTrainThickVolumeDataset(ThickVolumeDataset):
    def __init__(self, mask_omega_path, mask_subset_1_path, mask_subset_2_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.thick_array = self.thick_array.type(torch.complex64)
        self.mask_under = np.array(sio.loadmat(mask_omega_path)['mask'])
        self.mask_under = np.stack((self.mask_under, self.mask_under), axis=-1)
        self.mask_omega = torch.from_numpy(self.mask_under).float()

    def __len__(self):
        return self.end - self.start - 2

    def __getitem__(self, index):
        slice_1 = torch.index_select(self.thick_array, 3,
                                           torch.tensor([index + self.start])).squeeze(dim=3)
        slice_3 = torch.index_select(self.thick_array, 3,
                                           torch.tensor([index + self.start + 1])).squeeze(dim=3)
        slice_5 = torch.index_select(self.thick_array, 3,
                                           torch.tensor([index + self.start + 2])).squeeze(dim=3)

        return slice_1, slice_3, slice_5, self.mask_omega

class JointInferenceThickVolumeDataset(ThickVolumeDataset):
    def __init__(self, mask_omega_path, mask_subset_1_path, mask_subset_2_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.array = self.array.type(torch.complex64)
        self.thick_array = self.thick_array.type(torch.complex64)
        self.mask_under = np.array(sio.loadmat(mask_omega_path)['mask'])
        self.mask_under = np.stack((self.mask_under, self.mask_under), axis=-1)
        self.mask_omega = torch.from_numpy(self.mask_under).float()

    def __len__(self):
        return self.end - self.start - 1

    def __getitem__(self, index):
        input_slice_1 = torch.index_select(self.thick_array, 3,
                                           torch.tensor([index + self.start])).squeeze(dim=3)
        input_slice_2 = torch.index_select(self.thick_array, 3,
                                           torch.tensor([index + self.start + 1])).squeeze(dim=3)
        target_slice = torch.index_select(self.array, 3,
                                          torch.tensor([(index + self.start) * 2 + 1])).squeeze(dim=3)

        return input_slice_1, input_slice_2, target_slice, self.mask_omega


def get_volume_path_from_tsv(tsv_path, data_path):
    subs = pd.read_csv(tsv_path, sep='\t')
    subs = subs[(subs['original_study'] == 'ADNI3') & (subs['diagnosis_sc'] == 'CN')]  # filtering by condition

    mri_volume_paths = []

    for idx, sub in subs.iterrows():
        sub_id = sub['participant_id']
        sub_data_path = os.path.join(data_path, sub_id, f'ses-M00/anat/{sub_id}_ses-M00_T1w.nii.gz')
        # filter out subs without mri data modal
        if os.path.isfile(sub_data_path):
            mri_volume_paths.append(sub_data_path)

    return mri_volume_paths


def get_volume_datasets(tsv_path, data_path, ds_class, ds_kwargs, sub_limit=-1):

    mri_volume_paths = get_volume_path_from_tsv(tsv_path, data_path)

    dss = []
    for mri_volume_path in tqdm(mri_volume_paths):
        ds_kwargs.update({'volume':mri_volume_path})
        dss.append(
            ds_class(**ds_kwargs)
        )
        ds_kwargs.pop('volume')
        if len(dss) == sub_limit:  # confine number of subjects
            break

    ds = torch.utils.data.ConcatDataset(dss)

    return ds


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def unpack(sample):
        return [x[0].numpy() for x in sample]






    recon_thick_v_ds_kwargs = {
        'mask_omega_path': './mask/undersampling_mask/mask_8.00x_acs24.mat',
        'mask_subset_1_path': 'mask/selecting_mask/mask_2.00x_acs16.mat',
        'mask_subset_2_path': 'mask/selecting_mask/mask_2.50x_acs16.mat',
        'pad': (256, 256, 256),
        'q': 0.2,
    }
    recon_thick_v_ds = get_volume_datasets('./participants.tsv',
                                           '/mnt/d/data/ADNI/ADNIRawData',
                                           ReconThickVolumeDataset,
                                           recon_thick_v_ds_kwargs,
                                           sub_limit=1)

    os.makedirs('./test_viz', exist_ok=True)
    for i, sample in enumerate(recon_thick_v_ds):
        to_save = np.concatenate(unpack(sample), axis=1)
        plt.imsave(f'./test_viz/recon_thick_v_ds_{i}.png', to_save, cmap='gray')




    inter_pre_thick_v_ds_kwargs = {
        'pad': (256, 256, 256),
        'q': 0.2,
        'dim': 1
    }
    inter_pre_thick_v_ds = get_volume_datasets('./participants.tsv',
                                           '/mnt/d/data/ADNI/ADNIRawData',
                                           InterPretrainThickVolumeDataset,
                                           inter_pre_thick_v_ds_kwargs,
                                           sub_limit=1)

    os.makedirs('./inter_pre_thick_v_ds', exist_ok=True)
    for i, sample in enumerate(inter_pre_thick_v_ds):
        to_save = np.concatenate(unpack(sample), axis=1)
        plt.imsave(f'./inter_pre_thick_v_ds/{i}.png', to_save, cmap='gray')


    inter_inf_thick_v_ds_kwargs = {
        'pad': (256, 256, 256),
        'q': 0.2,
    }
    inter_pre_thick_v_ds = get_volume_datasets('./participants.tsv',
                                           '/mnt/d/data/ADNI/ADNIRawData',
                                           InterInferenceThickVolumeDataset,
                                           inter_inf_thick_v_ds_kwargs,
                                           sub_limit=1)

    os.makedirs('./inter_inf_thick_v_ds', exist_ok=True)
    for i, sample in enumerate(inter_pre_thick_v_ds):
        to_save = np.concatenate(unpack(sample), axis=1)
        plt.imsave(f'./inter_inf_thick_v_ds/{i}.png', to_save, cmap='gray')


    joint_inf_thick_v_ds_kwargs = {
        'mask_omega_path': './mask/undersampling_mask/mask_8.00x_acs24.mat',
        'mask_subset_1_path': 'mask/selecting_mask/mask_2.00x_acs16.mat',
        'mask_subset_2_path': 'mask/selecting_mask/mask_2.50x_acs16.mat',
        'pad': (256, 256, 256),
        'q': 0.2,
    }
    joint_inf_thick_v_ds = get_volume_datasets('./participants.tsv',
                                           '/mnt/d/data/ADNI/ADNIRawData',
                                           JointInferenceThickVolumeDataset,
                                           joint_inf_thick_v_ds_kwargs,
                                           sub_limit=1)

    os.makedirs('./joint_inf_thick_v_ds', exist_ok=True)
    for i, sample in enumerate(joint_inf_thick_v_ds):
        to_save = np.concatenate(unpack(sample), axis=1)
        plt.imsave(f'./joint_inf_thick_v_ds/{i}.png', to_save, cmap='gray')