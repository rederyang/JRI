# System / Python
import os
import argparse
import logging
import random
# tool
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
# Custom
from rec_net import make_model
from data import *
from utils import create_logger

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
# parameters related to model
parser.add_argument('--model', type=str, required=True, help='type of model')
parser.add_argument('--use-init-weights', '-uit', type=bool, default=True,
                    help='whether initialize model weights with defined types')
parser.add_argument('--net_G', type=str, default='DRDN', help='generator network')  # DRDN / SCNN
parser.add_argument('--n_recurrent', type=int, default=2, help='Number of reccurent block in model')
parser.add_argument('--use_prior', default=False, action='store_true', help='use prior')  # True / False
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
# learning rate, batch size, and etc
parser.add_argument('--seed', type=int, default=30, help='random seed number')
parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=2, help='batch size of single gpu')
parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--warmup-epochs', type=int, default=-1, help='number of warmup epochs')
parser.add_argument('--num-epochs', type=int, default=500, help='maximum number of epochs')
parser.add_argument('--reduce-lr-patience', type=int, default=100000)
parser.add_argument('--early-stop-patience', type=int, default=100000)
# parameters related to data and masks
parser.add_argument('--train-tsv-path', metavar='/path/to/training_data', default="./train_participants.tsv", type=str)
parser.add_argument('--val-tsv-path', metavar='/path/to/validation_data', default="./val_participants.tsv", type=str)
parser.add_argument('--test-tsv-path', metavar='/path/to/test_data', default="./test_participants.tsv", type=str)
parser.add_argument('--data-path', metavar='/path/to/data', default='/mnt/d/data/ADNI/ADNIRawData', type=str)
parser.add_argument('--train-obj-limit', type=int, default=20, help='number of objects in training set')
parser.add_argument('--val-obj-limit', type=int, default=5, help='number of objects in val set')
parser.add_argument('--test-obj-limit', type=int, default=20, help='number of objects in test set')
parser.add_argument('--drop-rate', type=float, default=0., help='ratio to drop edge slices')
parser.add_argument('--u-mask-path', type=str, default='./mask/undersampling_mask/mask_8.00x_acs24.mat',
                    help='undersampling mask')
parser.add_argument('--s-mask-up-path', type=str, default='./mask/selecting_mask/mask_2.00x_acs16.mat',
                    help='selection mask in up network')
parser.add_argument('--s-mask-down-path', type=str, default='./mask/selecting_mask/mask_2.50x_acs16.mat',
                    help='selection mask in down network')
# parser.add_argument('--prefetch', action='store_false')
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
    model.load_best()
    model = model.cuda()

    logger.info('Current epoch {}.'.format(model.epoch))
    logger.info('Current best metric in train phase is {}.'.format(model.best_target_metric))

    ds_kwargs = {
        'mask_omega_path': args.u_mask_path,
        'mask_subset_1_path': args.s_mask_up_path,
        'mask_subset_2_path': args.s_mask_down_path,
        'pad': (256, 256, 256),
        'q': args.drop_rate,
    }

    logger.info(ds_kwargs)

    # read nii paths to recon
    mri_volume_paths = get_volume_path_from_tsv(args.train_tsv_path, args.data_path)
    mri_volume_paths = mri_volume_paths[:args.train_obj_limit]

    model.eval()

    for nii_path in mri_volume_paths:
        # construct dataset and dataloader
        ds_kwargs.update({'volume': nii_path})
        volume_dataset = ReconVolumeDataset(**ds_kwargs)
        dataloader = DataLoader(dataset=volume_dataset, batch_size=args.batch_size, shuffle=False)

        # recon slice by slice
        rec_volume = []
        for item in iter(tqdm(dataloader)):
            model.set_input('test', item)
            output_i_1, output_i_2, _ = model.inference()
            output_i_1 = \
            torch.abs(torch.view_as_complex(output_i_1.permute(0, 2, 3, 1).contiguous())).detach().cpu().numpy()
            rec_volume.append(output_i_1)
        rec_volume = np.concatenate(rec_volume, axis=0).transpose((1, 2, 0))

        # recover to nii
        nii = nib.load(nii_path)  # original nii
        array = nib.as_closest_canonical(nii).get_fdata().astype(np.float32)  # original array
        rec_volume = crop_np(rec_volume, array.shape)  # recover shape
        rec_volume = rec_volume * np.max(array)  # recover value range
        rec_nii = nib.Nifti1Image(rec_volume, nii.affine)  # affine information from original nii

        # save to nii.gz
        rec_nii_path = nii_path.replace('ADNIRawData', 'ADNIRecData')
        os.makedirs(''.join(os.path.split(rec_nii_path)[:-1]), exist_ok=True)
        nib.save(rec_nii, rec_nii_path)

        logger.info(f'Reconstruction of {nii_path} is complete.')
        logger.info(f'Saved to {rec_nii_path}.')


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
