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
parser.add_argument('--reduce-lr-patience', type=int, default=100000)
parser.add_argument('--early-stop-patience', type=int, default=100000)
parser.add_argument('--patience', type=int, default=100000)
# parameters related to data and masks
parser.add_argument('--train-tsv-path', metavar='/path/to/training_data', default="./train_participants.tsv", type=str)
parser.add_argument('--val-tsv-path', metavar='/path/to/validation_data', default="./val_participants.tsv", type=str)
parser.add_argument('--test-tsv-path', metavar='/path/to/test_data', default="./test_participants.tsv", type=str)
parser.add_argument('--data-path', metavar='/path/to/data', default='/mnt/d/data/ADNI/ADNIRawData', type=str)
parser.add_argument('--rec-data-path', metavar='/path/to/data', default='/mnt/d/data/ADNI/ADNIRecData', type=str)
parser.add_argument('--use-rec-data', action='store_true', help='whether to use recon data to train')
parser.add_argument('--train-obj-limit', type=int, default=20, help='number of objects in training set')
parser.add_argument('--val-obj-limit', type=int, default=5, help='number of objects in val set')
parser.add_argument('--test-obj-limit', type=int, default=20, help='number of objects in test set')
parser.add_argument('--drop-rate', type=float, default=0.2, help='ratio to drop edge slices')
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
    if args.mode == 'test':
        model.load_best()
    elif args.resume:
        model.load()

    logger.info('Current epoch {}.'.format(model.epoch))
    logger.info('Current learning rate is {}.'.format(model.optimizer.param_groups[0]['lr']))
    logger.info('Current best metric in train phase is {}.'.format(model.best_target_metric))

    inter_thick_v_ds_kwargs = {
        'pad': (256, 256, 256),
        'q': args.drop_rate,
    }

    if args.mode == 'test':
        # data
        test_ds = get_volume_datasets(args.test_tsv_path,
                                      args.data_path,
                                      InterInferenceThickVolumeDataset,
                                      inter_thick_v_ds_kwargs,
                                      sub_limit=args.test_obj_limit)
        logger.info(f'Testing dataset size: \t'
                    f'number of subjects: {args.test_obj_limit}\t'
                    f'size of dataset: {len(test_ds)}\t')
        test_loader = DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        test_loader = tqdm(test_loader, desc='testing', total=int(len(test_loader)))

        # run one epoch
        test_log = model.test_one_epoch(test_loader)

        # log
        logger.info(f'time:{test_log["time"]:.5f}s\t'
                    f'test_loss:{test_log["loss"]:.7f}\t'
                    f'test_psnr:{test_log["psnr"]:.5f}\t'
                    f'test_ssim:{test_log["ssim"]:.5f}\t')

        return

    # data
    # inter_thick_v_ds_kwargs.update({'dim': 1})
    # train_set_1 = get_volume_datasets(args.train_tsv_path,
    #                                 args.rec_data_path if args.use_rec_data else args.data_path,
    #                                 InterPretrainThickVolumeDataset,
    #                                 inter_thick_v_ds_kwargs,
    #                                 sub_limit=args.train_obj_limit)
    # inter_thick_v_ds_kwargs.update({'dim': 2})
    # train_set_2 = get_volume_datasets(args.train_tsv_path,
    #                                 args.rec_data_path if args.use_rec_data else args.data_path,
    #                                 InterPretrainThickVolumeDataset,
    #                                 inter_thick_v_ds_kwargs,
    #                                 sub_limit=args.train_obj_limit)
    # train_set = torch.utils.data.ConcatDataset([train_set_1, train_set_2])
    # inter_thick_v_ds_kwargs.pop('dim')
    train_set = get_volume_datasets(args.train_tsv_path,
                                    args.data_path,
                                    InterInferenceThickVolumeDataset,
                                    inter_thick_v_ds_kwargs,
                                    sub_limit=args.train_obj_limit)
    val_set = get_volume_datasets(args.val_tsv_path,
                                  args.data_path,
                                  InterInferenceThickVolumeDataset,
                                  inter_thick_v_ds_kwargs,
                                  sub_limit=args.val_obj_limit)

    logger.info(f'Training dataset size: \t'
                f'number of subjects: {args.train_obj_limit}\t'
                f'size of pretraining dataset: {len(train_set)}\t')
    pre_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    logger.info(f'Validation dataset size: \t'
                f'number of subjects: {args.val_obj_limit}\t'
                f'size of validation dataset: {len(val_set)}\t')
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # pre-training
    for epoch in range(model.epoch + 1, args.pre_epochs + 1):
        # data and run one epoch
        train_loader = tqdm(pre_loader, desc='pre-training', total=int(len(pre_loader)))
        train_log = model.train_one_epoch('pre', train_loader)
        val_loader = tqdm(val_loader, desc='valing', total=int(len(val_loader)))
        val_log = model.eval_one_epoch(val_loader)

        # output log
        logger.info(f'epoch:{train_log["epoch"]:<8d}\t'
                    f'time:{train_log["time"]:.2f}s\t'
                    f'lr:{train_log["lr"]:.8f}\t'
                    f'train_loss:{train_log["loss"]:.7f}\t'
                    f'val_loss:{val_log["loss"]:.7f}\t'
                    f'val_psnr:{val_log["psnr"]:.5f}\t'
                    f'val_ssim:{val_log["ssim"]:.5f}\t')

        if model.signal_to_stop:
            logger.info('The experiment is early stop!')
            break

    # self-training
    # for epoch in range(model.epoch + 1, args.pre_epochs + args.self_epochs + 1):
    #     # data and run one epoch
    #     train_loader = tqdm(zip(pre_loader, cyc_loader), desc='self-training', total=int(len(cyc_loader)))
    #     train_log = model.train_one_epoch('self', train_loader)
    #     val_loader = tqdm(val_loader, desc='valing', total=int(len(val_loader)))
    #     val_log = model.eval_one_epoch(val_loader)
    #
    #     # output log
    #     logger.info(f'epoch:{train_log["epoch"]:<8d}\t'
    #                 f'time:{train_log["time"]:.2f}s\t'
    #                 f'lr:{train_log["lr"]:.8f}\t'
    #                 f'train_loss:{train_log["loss"]:.7f}\t'
    #                 f'val_loss:{val_log["loss"]:.7f}\t'
    #                 f'val_psnr:{val_log["psnr"]:.5f}\t'
    #                 f'val_ssim:{val_log["ssim"]:.5f}\t')
    #
    #     if model.signal_to_stop:
    #         logger.info('The experiment is early stop!')
    #         break


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
