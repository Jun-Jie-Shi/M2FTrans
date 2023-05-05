#!/usr/bin/env python3
# encoding: utf-8
import os
import random
import torch
import torch.backends.cudnn as cudnn
import warnings
import numpy as np
import math
import logging
import time

# h_crop = 160
# w_crop = 192
# d_crop = 128
h_crop = 128
w_crop = 128
d_crop = 128

def init_env(gpu_id='0', seed=1037):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    warnings.filterwarnings('ignore')

class LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, warmup=100, mode='poly'):
        self.mode = mode
        self.lr = base_lr
        self.num_epochs = num_epochs
        self.warmup = warmup

    def __call__(self, optimizer, epoch):
        if self.mode == 'poly':
            now_lr = round(self.lr * np.power(1 - np.float32(epoch)/np.float32(self.num_epochs), 0.9), 8)
        elif self.mode == 'warmup':
            if epoch < self.warmup*2:
                now_lr = round(0.5 * self.lr * (1.0 + math.cos(((np.float32(epoch)/np.float32(self.warmup))*math.pi))),8)
            else:
                now_lr = round(self.lr * np.power(1 - (np.float32(epoch) - np.float32(self.warmup*2))/(np.float32(self.num_epochs)-np.float32(self.warmup*2)), 0.9), 8)
        elif self.mode == 'cousinewarmup':
            if self.warmup == 0:
                if epoch < 100:
                    now_lr = round(self.lr * (math.sin(((np.float32(epoch))/(np.float32(100.0 * 2.0)))*math.pi)),8)
                else:
                    now_lr = round(0.5 * self.lr * (1.0 + math.cos(((np.float32(epoch) - np.float32(100.0))/(np.float32(self.num_epochs)-np.float32(100.0)))*math.pi)), 8)
            else:
                if epoch < self.warmup*2:
                    now_lr = round(0.5 * self.lr * (1.0 + math.cos(((np.float32(epoch)/np.float32(self.warmup))*math.pi))),8)
                else:
                    now_lr = round(0.5 * self.lr * (1.0 + math.cos(((np.float32(epoch) - np.float32(self.warmup*2))/(np.float32(self.num_epochs)-np.float32(self.warmup*2)))*math.pi)), 8)
        elif self.mode == 'warmuppoly':
            if epoch < 50:
                now_lr = round(self.lr * (((np.float32(epoch+1.0))/(np.float32(50.0)))),8)
            else:
                now_lr = round(self.lr * np.power(1 - (np.float32(epoch) - np.float32(50.0))/(np.float32(self.num_epochs+1)-np.float32(50.0)), 0.9), 8)
        self._adjust_learning_rate(optimizer, now_lr)
        return now_lr

    def _adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def softmax_output_dice_class4(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 4).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3 
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def test_softmax(
        test_loader,
        model,
        dataname = '/home/sjj/M2FTrans/BraTS/BRATS2018',
        feature_mask=None,
        mask_name=None):

    H, W, D = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, h_crop, w_crop, d_crop).float().cuda()

    if dataname in ['/home/sjj/M2FTrans/BraTS/BRATS2020', '/home/sjj/M2FTrans/BraTS/BRATS2018']:
        num_cls = 4
        class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'


    for i, data in enumerate(test_loader):
        target = data[1].cuda()
        x = data[0].cuda()
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda()
        _, _, H, W, D = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = np.int_(np.ceil((H - h_crop) / (h_crop * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int_(h_crop * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - h_crop)

        w_cnt = np.int_(np.ceil((W - w_crop) / (w_crop * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int_(w_crop * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - w_crop)

        d_cnt = np.int_(np.ceil((D - d_crop) / (d_crop * (1 - 0.5))))
        d_idx_list = range(0, d_cnt)
        d_idx_list = [d_idx * np.int_(d_crop * (1 - 0.5)) for d_idx in d_idx_list]
        d_idx_list.append(D - d_crop)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, D).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for d in d_idx_list:
                    weight1[:, :, h:h+h_crop, w:w+w_crop, d:d+d_crop] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, D).float().cuda()
        model.module.is_training=False

        for h in h_idx_list:
            for w in w_idx_list:
                for d in d_idx_list:
                    x_input = x[:, :, h:h+h_crop, w:w+w_crop, d:d+d_crop]
                    pred_part = model(x_input, mask)
                    pred[:, :, h:h+h_crop, w:w+w_crop, d:d+d_crop] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :D]
        pred = torch.argmax(pred, dim=1)


        scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)

        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print (msg)
    logging.info(msg)
    model.train()
    return vals_evaluation.avg



def test_center(
        test_loader,
        model,
        feature_mask=None,
        mask_name=None):

    # H, W, D = 240, 240, 155
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()

    class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
    class_separate = 'ncr_net', 'edema', 'enhancing'


    for batch_id, (batch_x, batch_y, patient_ids) in enumerate(test_loader):
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        mask = torch.from_numpy(feature_mask)
        batchsize = batch_x.size(0)
        mask = mask.repeat(batchsize, 1)
        mask = mask.cuda()
        model.module.is_training=False
        pred = model(batch_x, mask)
        pred = torch.argmax(pred, dim=1)

        # logging.info(batch_y.size())

        scores_separate, scores_evaluation = softmax_output_dice_class4(pred, batch_y)


        for k, patient_id in enumerate(patient_ids):
            msg = 'Subject {}/{}, {}/{}'.format((batch_id+1), len(test_loader), (k+1), len(patient_ids))
            msg += '{:>20}, '.format(patient_id)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            # logging.info(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])

            logging.info(msg)
    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print (msg)
    logging.info(msg)
    model.train()
    return vals_evaluation.avg