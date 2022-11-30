# System / Python
import argparse
import os
import random

# tool
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import get_volume_datasets
# Custom
from inter_net import make_model
from data import *
from utils import create_logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
# parameters related to model
parser.add_argument('--model', type=str, default='TSCNet', help='type of model')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
parser.add_argument('--c-in', type=int, default=1)
parser.add_argument('--c-feat', type=int, default=64)  # 32
parser.add_argument('--blocks', type=int, default=3)  # 3
parser.add_argument('--layers', type=int, default=5)  # 3
parser.add_argument('--grow-rate', type=int, default=64)  # 32
parser.add_argument('--adv-weight', type=float, default=0.1, help='adversarial training loss weight')
# learning rate, batch size, and etc
parser.add_argument('--seed', type=int, default=30, help='random seed number')
parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=2, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--warmup-epochs', type=int, default=-1, help='number of warmup epochs')
parser.add_argument('--num-epochs', type=int, default=500, help='maximum number of epochs')
parser.add_argument('--pre-epochs', type=int, default=100)
parser.add_argument('--self-epochs', type=int, default=150)
parser.add_argument('--patience', type=int, default=100000)
# parameters related to data and masks
parser.add_argument('--train-tsv-path', metavar='/path/to/training_data', default="./train_participants.tsv", type=str)
parser.add_argument('--val-tsv-path', metavar='/path/to/validation_data', default="./val_participants.tsv", type=str)
parser.add_argument('--test-tsv-path', metavar='/path/to/test_data', default="./test_participants.tsv", type=str)
parser.add_argument('--data-path', metavar='/path/to/data', default='/mnt/d/data/ADNI/ADNIRawData', type=str)
parser.add_argument('--rec-data-path', metavar='/path/to/data', default='/mnt/d/data/ADNI/ADNIRecData', type=str)
parser.add_argument('--use-rec-data', action='store_true', help='whether to use recon data')
parser.add_argument('--train-obj-limit', type=int, default=20, help='number of objects in training set')
parser.add_argument('--val-obj-limit', type=int, default=5, help='number of objects in val set')
parser.add_argument('--test-obj-limit', type=int, default=20, help='number of objects in test set')
parser.add_argument('--drop-rate', type=int, default=0., help='ratio to drop edge slices')
# save path
parser.add_argument('--output-path', type=str, default='./runs/test_run/', help='output path')
# others
parser.add_argument('--mode', '-m', type=str, default='train',
                    help='whether training or test model, value should be set to train or test')
parser.add_argument('--resume', action='store_true', help='whether resume to train')
parser.add_argument('--save-evaluation-viz', action='store_true')


def solvers(args):
    # logger and devices
    logger = create_logger(args)

    logger.info(args)

    # model
    model = make_model(0, args)  # 0 is rank
    model = model.cuda()  # 1. to device 2. load optimizer
    model.load_best()

    logger.info('Current epoch {}.'.format(model.epoch))
    logger.info('Current best metric in train phase is {}.'.format(model.best_target_metric))

    ds_kwargs = {
        'pad': (256, 256, 256),
        'q': args.drop_rate,
    }

    logger.info(ds_kwargs)

    mri_volume_paths = get_volume_path_from_tsv(args.train_tsv_path, args.rec_data_path if args.use_rec_data else args.data_path)
    mri_volume_paths = mri_volume_paths[:args.train_obj_limit]

    model.eval()

    for nii_path in mri_volume_paths:
        # construct dataset and dataloader
        ds_kwargs.update({'volume': nii_path})
        volume_dataset = InterInferenceThickVolumeDataset(**ds_kwargs)
        dataloader = DataLoader(dataset=volume_dataset, batch_size=args.batch_size, shuffle=False)

        # recon slice by slice
        triplet_slices = []
        for item in iter(tqdm(dataloader)):
            model.set_input('test', item)
            middle_slice, _ = model.inference()
            upper_slice = model.input_slice_1.detach().cpu().numpy()
            middle_slice = middle_slice.detach().cpu().numpy()
            lower_slice = model.input_slice_2.detach().cpu().numpy()
            triplet_slice = np.concatenate([upper_slice, middle_slice, lower_slice], axis=1)  # cat at channel dim
            triplet_slices.append(triplet_slice)  # [[bs, 3, x, y], ...]
        triplet_slices = np.concatenate(triplet_slices, axis=0)  # [N, 3, x, y], N == size of ds

        volume = []
        for i in range(triplet_slices.shape[0]):
            volume.append(triplet_slices[i, 0])
            volume.append(triplet_slices[i, 1])
        volume.append(triplet_slices[-1, -1])  # the last lower slice within the last triplet slice group

        # FIXME: the result might miss one slice at bottom, here we add one zero-valued slice
        volume.append(np.zeros_like(volume[-1]))

        inter_volume = np.stack(volume, axis=0).transpose((1, 2, 0))

        # recover to nii
        nii = nib.load(nii_path)  # original nii
        array = nib.as_closest_canonical(nii).get_fdata().astype(np.float32)  # original array
        inter_volume = crop_np(inter_volume, array.shape)  # recover shape
        inter_volume = inter_volume * np.max(array)  # recover value range
        inter_nii = nib.Nifti1Image(inter_volume, nii.affine)  # affine information from original nii

        # save to nii.gz
        inter_nii_path = nii_path.replace('ADNIRecData', 'ADNIInterData')
        inter_nii_path = inter_nii_path.replace('ADNIRawData', 'ADNIInterData')
        os.makedirs(''.join(os.path.split(inter_nii_path)[:-1]), exist_ok=True)
        nib.save(inter_nii, inter_nii_path)

        logger.info(f'Interpolation of {nii_path} is complete.')
        logger.info(f'Saved to {inter_nii_path}.')


def main():
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    args.model_save_path = os.path.join(args.output_path, 'checkpoints')
    args.log_path = os.path.join(args.output_path, 'log.txt')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    solvers(args)


if __name__ == '__main__':
    main()
