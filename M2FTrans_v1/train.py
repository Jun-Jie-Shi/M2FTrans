#coding=utf-8
import argparse
import logging
import os
import random
import time
from collections import OrderedDict
import csv

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
from data.data_utils import init_fn
from data.datasets_nii import (Brats_loadall_nii, Brats_loadall_test_nii,
                               Brats_loadall_val_nii)
from data.transforms import *
# from visualizer import get_local
# get_local.activate()
from models import rfnet, mmformer, fusiontrans
from predict import AverageMeter, test_softmax, test_dice_hd95_softmax
# from predict import AverageMeter, test_softmax, test_softmax_visualize
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import Parser, criterions
from utils.lr_scheduler import LR_Scheduler, MultiEpochsDataLoader
from utils.parser import setup
# from utils.visualize import visualize_heads

parser = argparse.ArgumentParser()

parser.add_argument('--model', default='fusiontrans', type=str)
parser.add_argument('-batch_size', '--batch_size', default=4, type=int, help='Batch size')
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--dataname', default='/home/sjj/M2FTrans/BraTS/BRATS2020', type=str)
parser.add_argument('--datapath', default='/home/sjj/M2FTrans/BraTS/BRATS2020_Training_none_npy', type=str)
parser.add_argument('--savepath', default=None, type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--pretrain', default=None, type=str)
parser.add_argument('--region_fusion_start_epoch', default=0, type=int)
parser.add_argument('--seed', default=1037, type=int)
parser.add_argument('--needvalid', default=False, type=bool)
parser.add_argument('--csvpath', default='/home/sjj/M2FTrans/csv/', type=str)
path = os.path.dirname(__file__)

## parse arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

csvpath = args.csvpath
os.makedirs(csvpath, exist_ok=True)

###tensorboard writer
writer = SummaryWriter(os.path.join(args.savepath, 'summary'))

###modality missing mask
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_torch.int())

# masks_valid = [[False, False, True, False],
#             [False, True, True, False],
#             [True, True, False, True],
#             [True, True, True, True]]
masks_valid = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
# t1,t1cet1,flairticet2,flairt1cet1t2
masks_valid_torch = torch.from_numpy(np.array(masks_valid))
masks_valid_array = np.array(masks_valid)
masks_all = [True, True, True, True]
masks_all_torch = torch.from_numpy(np.array(masks_all))
# mask_name_valid = ['t1',
#                 't1cet1',
#                 'flairt1cet2',
#                 'flairt1cet1t2']
mask_name_valid = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
print (masks_valid_torch.int())

def main():
    ##########setting seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    ##########setting models
    if args.dataname in ['/home/sjj/MMMSeg/BraTS/BRATS2021', '/home/sjj/MMMSeg/BraTS/BRATS2020', '/home/sjj/MMMSeg/BraTS/BRATS2018']:
        num_cls = 4
    else:
        print ('dataset is error')
        exit(0)

    if args.model == 'fusiontrans':
        model = fusiontrans.Model(num_cls=num_cls)
    elif args.model == 'rfnet':
        model = rfnet.Model(num_cls=num_cls)
    elif args.model == 'mmformer':
        model = mmformer.Model(num_cls=num_cls)

    print (model)
    model = torch.nn.DataParallel(model).cuda()

    ##########Setting learning schedule and optimizer
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs, warmup=args.region_fusion_start_epoch, mode='warmuppoly')
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
    optimizer = torch.optim.AdamW(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    ##########Setting data
        ####BRATS2020
    if args.dataname == '/home/sjj/M2FTrans/BraTS/BRATS2020':
        train_file = '/home/sjj/M2FTrans/BraTS/BRATS2020_Training_none_npy/train.txt'
        test_file = '/home/sjj/M2FTrans/BraTS/BRATS2020_Training_none_npy/test.txt'
        # valid_file = '/home/sjj/M2FTrans/BraTS/BRATS2020_Training_none_npy/val.txt'
    elif args.dataname == '/home/sjj/MMMSeg/BraTS/BRATS2021':
        ####BRATS2021
        train_file = '/home/sjj/MMMSeg/BraTS/BRATS2021_Training_none_npy/train.txt'
        test_file = '/home/sjj/MMMSeg/BraTS/BRATS2021_Training_none_npy/test.txt'
        # valid_file = '/home/sjj/MMMSeg/BraTS/BRATS2021_Training_none_npy/val.txt'
    elif args.dataname == '/home/sjj/M2FTrans/BraTS/BRATS2018':
        ####BRATS2018 contains three splits (1,2,3)
        train_file = '/home/sjj/M2FTrans/BraTS/BRATS2018_Training_none_npy/train1.txt'
        test_file = '/home/sjj/M2FTrans/BraTS/BRATS2018_Training_none_npy/test1.txt'
        # valid_file = '/home/sjj/M2FTrans/BraTS/BRATS2018_Training_none_npy/val.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=train_file)
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=test_file)
    # valid_set = Brats_loadall_val_nii(transforms=args.train_transforms, root=args.datapath, num_cls=num_cls, train_file=valid_file)
    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn)
    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    # valid_loader = MultiEpochsDataLoader(
    #     dataset=valid_set,
    #     batch_size=args.batch_size,
    #     num_workers=8,
    #     pin_memory=True,
    #     shuffle=True,
    #     worker_init_fn=init_fn)

    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # logging.info('pretrained_dict: {}'.format(pretrained_dict))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info('load ok')


    ##########Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    iter_per_epoch = args.iter_per_epoch
    train_iter = iter(train_loader)
    # valid_iter = iter(valid_loader)
    for epoch in range(args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b = time.time()
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)
            x, target, mask = data[:3]
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            batchsize = mask.size(0)
            # all modalities test
            # mask = masks_all_torch.repeat(batchsize, 1)
            mask = mask[0].repeat(batchsize, 1)  ## to be test

            mask = mask.cuda(non_blocking=True)

            model.module.is_training = True

            fuse_pred, sep_preds, prm_preds = model(x, mask)

            ###Loss compute
            fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
                sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
            sep_loss = sep_cross_loss + sep_dice_loss


            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
                prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
            prm_loss = prm_cross_loss + prm_dice_loss


            if epoch < args.region_fusion_start_epoch:
                loss = fuse_loss * 0.0 + sep_loss + prm_loss * 0.0
            else:
                loss = fuse_loss + sep_loss + prm_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ###log
            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.num_epochs, (i+1), iter_per_epoch, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())

            logging.info(msg)
        b_train = time.time()
        logging.info('train time per epoch: {}'.format(b_train - b))

        #########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            },
            file_name)

        if (epoch+1) % 50 == 0 or (epoch>=(args.num_epochs-10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)
        # #########Validate this epoch model
        # if (args.needvalid == True):
        #     with torch.no_grad():
        #         logging.info('#############validation############')
        #         score_modality = torch.zeros(16)
        #         for j, masks in enumerate(masks_valid_array):
        #             logging.info('{}'.format(mask_name_valid[j]))
        #             for i in range(len(valid_loader)):
        #             # step = (i+1) + epoch*iter_per_epoch
        #             ###Data load
        #                 try:
        #                     data = next(valid_iter)
        #                 except:
        #                     valid_iter = iter(valid_loader)
        #                     data = next(valid_iter)
        #                 x, target= data[:2]
        #                 x = x.cuda(non_blocking=True)
        #                 target = target.cuda(non_blocking=True)
        #                 batchsize=x.size(0)


        #                 mask = torch.unsqueeze(torch.from_numpy(masks), dim=0)
        #                 mask = mask.repeat(batchsize,1)
        #                 mask = mask.cuda(non_blocking=True)

        #                 model.module.is_training = True
        #                 # fuse_pred, sep_preds, prm_preds = model(x, mask)

        #                 fuse_pred, sep_preds, prm_preds = model(x, mask)

        #                 ###Loss compute
        #                 fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
        #                 fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
        #                 fuse_loss = fuse_cross_loss + fuse_dice_loss

        #                 sep_cross_loss = torch.zeros(1).cuda().float()
        #                 sep_dice_loss = torch.zeros(1).cuda().float()
        #                 for sep_pred in sep_preds:
        #                     sep_cross_loss += criterions.softmax_weighted_loss(sep_pred, target, num_cls=num_cls)
        #                     sep_dice_loss += criterions.dice_loss(sep_pred, target, num_cls=num_cls)
        #                 sep_loss = sep_cross_loss + sep_dice_loss

        #                 prm_cross_loss = torch.zeros(1).cuda().float()
        #                 prm_dice_loss = torch.zeros(1).cuda().float()
        #                 for prm_pred in prm_preds:
        #                     prm_cross_loss += criterions.softmax_weighted_loss(prm_pred, target, num_cls=num_cls)
        #                     prm_dice_loss += criterions.dice_loss(prm_pred, target, num_cls=num_cls)
        #                 prm_loss = prm_cross_loss + prm_dice_loss

        #                 loss = fuse_loss + sep_loss + prm_loss

        #                 # loss = fuse_loss
        #                 # score -= loss
        #                 score_modality[j] -= loss.item()
        #                 score_modality[15] -= loss.item()
        #         score_modality[15] = score_modality[15] / len(masks_valid_array)
        #         if epoch == 0:
        #             best_score = score_modality[15]
        #             best_epoch = epoch
        #         elif score_modality[15] > best_score:
        #             best_score = score_modality[15]
        #             best_epoch = epoch
        #             file_name = os.path.join(ckpts, 'model_best.pth')
        #             torch.save({
        #                 'epoch': epoch,
        #                 'state_dict': model.state_dict(),
        #                 'optim_dict': optimizer.state_dict(),
        #                 },
        #                 file_name)

        #         for z, _ in enumerate(masks_valid_array):
        #             writer.add_scalar('{}'.format(mask_name_valid[z]), score_modality[z].item(), global_step=epoch+1)
        #         writer.add_scalar('score_average', score_modality[15].item(), global_step=epoch+1)
        #         logging.info('epoch total score: {}'.format(score_modality[15].item()))
        #         logging.info('best score: {}'.format(best_score.item()))
        #         logging.info('best epoch: {}'.format(best_epoch + 1))
        #         logging.info('validate time per epoch: {}'.format(time.time() - b_train))

    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Test the last epoch model
    # writer_visualize = SummaryWriter(log_dir="visualize/result")
    # visualize_step = 0
    test_dice_score = AverageMeter()
    test_hd95_score = AverageMeter()
    csv_name = os.path.join(csvpath, '{}.csv'.format(args.model))
    with torch.no_grad():
        logging.info('###########test last epoch model###########')
        file = open(csv_name, "a+")
        csv_writer = csv.writer(file)
        csv_writer.writerow(['WT Dice', 'TC Dice', 'ET Dice','ETPro Dice', 'WT HD95', 'TC HD95', 'ET HD95' 'ETPro HD95'])
        file.close()
        for i, mask in enumerate(masks_test[::-1]):
            logging.info('{}'.format(mask_name[::-1][i]))
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow([mask_name[::-1][i]])
            file.close()
            dice_score, hd95_score = test_dice_hd95_softmax(
                            test_loader,
                            model,
                            dataname = args.dataname,
                            feature_mask = mask,
                            mask_name = mask_name[::-1][i],
                            csv_name = csv_name,
                            )
            test_dice_score.update(dice_score)
            test_hd95_score.update(hd95_score)

        logging.info('Avg Dice scores: {}'.format(test_dice_score.avg))
        logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))


    # ##########Test the best epoch model
    # file_name = os.path.join(ckpts, 'model_best.pth')
    # checkpoint = torch.load(file_name)
    # logging.info('best epoch: {}'.format(checkpoint['epoch']+1))
    # model.load_state_dict(checkpoint['state_dict'])
    # test_best_score = AverageMeter()
    # with torch.no_grad():
    #     logging.info('###########test validate best model###########')
    #     for i, mask in enumerate(masks_test[::-1]):
    #         logging.info('{}'.format(mask_name[::-1][i]))
    #         dice_best_score = test_softmax(
    #                         test_loader,
    #                         model,
    #                         dataname = args.dataname,
    #                         feature_mask = mask,
    #                         mask_name = mask_name[::-1][i])
    #         test_best_score.update(dice_best_score)
    #     logging.info('Avg scores: {}'.format(test_best_score.avg))



if __name__ == '__main__':
    main()
