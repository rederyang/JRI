# System / Python
import os
import argparse
import logging
import random
# tool
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Custom
from rec_net import make_model
from data import *
from utils import create_logger, dict2line

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
parser.add_argument('--syn-data-path', metavar='/path/to/data', default='/mnt/d/data/ADNI/ADNIInterData', type=str)
parser.add_argument('--use-syn-data', action='store_true')
parser.add_argument('--train-obj-limit', type=int, default=20, help='number of objects in training set')
parser.add_argument('--val-obj-limit', type=int, default=5, help='number of objects in val set')
parser.add_argument('--test-obj-limit', type=int, default=20, help='number of objects in test set')
parser.add_argument('--drop-rate', type=float, default=0.2, help='ratio to drop edge slices')
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
    writer = SummaryWriter(log_dir=args.output_path)
    logger = create_logger(args)

    logger.info(args)

    # model
    model = make_model(0, args)  # 0 is rank
    if args.mode == 'test':
        model.load_best()
    elif args.resume:
        model.load()
    model = model.cuda()

    logger.info('Current epoch {}.'.format(model.epoch))
    logger.info('Current learning rate is {}.'.format(model.optimizer.param_groups[0]['lr']))
    logger.info('Current best metric in train phase is {}.'.format(model.best_target_metric))

    recon_thick_v_ds_kwargs = {
        'mask_omega_path': args.u_mask_path,
        'mask_subset_1_path': args.s_mask_up_path,
        'mask_subset_2_path': args.s_mask_down_path,
        'pad': (256, 256, 256),
        'q': args.drop_rate,
    }

    if args.mode == 'test':
        test_set = get_volume_datasets(args.test_tsv_path,
                                        args.data_path,
                                        ReconThickVolumeDataset,
                                        recon_thick_v_ds_kwargs,
                                        sub_limit=args.test_obj_limit)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        logger.info('The size of test dataset is {}.'.format(len(test_set)))
        test_loader = tqdm(test_loader, desc='testing', total=int(len(test_loader)))

        # run one epoch
        test_log = model.test_one_epoch(test_loader)
        logger.info(dict2line(test_log))

        return

    # data
    train_set = get_volume_datasets(args.train_tsv_path,
                                    args.syn_data_path if args.use_syn_data else args.data_path,
                                    ReconVolumeDataset if args.use_syn_data else ReconThickVolumeDataset,
                                    recon_thick_v_ds_kwargs,
                                    sub_limit=args.train_obj_limit)
    val_set = get_volume_datasets(args.val_tsv_path,
                                    args.data_path,
                                    ReconThickVolumeDataset,
                                    recon_thick_v_ds_kwargs,
                                    sub_limit=args.val_obj_limit)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    logger.info('The size of train dataset is {}.'.format(len(train_set)))
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    logger.info('The size of val dataset is {}.'.format(len(val_set)))

    # training loop
    for epoch in range(model.epoch + 1, args.num_epochs + 1):
        # data and run one epoch
        train_loader = tqdm(train_loader, desc='training', total=int(len(train_loader)))
        train_log = model.train_one_epoch(train_loader)
        logger.info(dict2line(train_log))
        for k in filter(lambda x: 'train' in x, train_log.keys()):
            writer.add_scalar('train/'+k, train_log[k], train_log['epoch'])

        val_loader = tqdm(val_loader, desc='valing', total=int(len(val_loader)))
        val_log = model.eval_one_epoch(val_loader)
        logger.info(dict2line(val_log))
        for k in filter(lambda x: 'val' in x, val_log.keys()):
            writer.add_scalar('val/'+k, val_log[k], val_log['epoch'])

        if model.signal_to_stop:
            logger.info('The experiment is early stop!')
            break


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
