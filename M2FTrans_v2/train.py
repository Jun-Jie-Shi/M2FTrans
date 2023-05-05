import yaml
from data import make_data_loaders
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import init_env, LR_Scheduler, AverageMeter, test_center
import nibabel as nib
import numpy as np
from models import  m2ftrans
import logging
import random
import time
import losses
from torch.utils.tensorboard import SummaryWriter


config = yaml.load(open('./config.yml'), Loader=yaml.FullLoader)

ckpts = config['path_to_ckpts']
os.makedirs(ckpts, exist_ok=True)

logging.basicConfig(level=logging.INFO,
        format='%(asctime)s %(message)s', filename='./logs/brats2018_m2ftrans.txt')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logging.getLogger('').addHandler(console)

writer = SummaryWriter(os.path.join(ckpts, 'summary'))

mask_array = np.array([[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]])
mask_name = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']

def main():
    init_env('2,3')
    model = m2ftrans.Model(num_cls=config['number_classes'])
    print (model)
    model = torch.nn.DataParallel(model).cuda()
    loaders = make_data_loaders(config)
    lr_schedule = LR_Scheduler(config['lr'], config['epochs'], warmup=0, mode='warmuppoly')
    train_params = [{'params': model.parameters(), 'lr': config['lr'], 'weight_decay':config['weight_decay']}]
    optimizer = torch.optim.AdamW(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    if config['path_to_resume'] is not None:
        checkpoint = torch.load(config['path_to_resume'])
        pretrained_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # logging.info('pretrained_dict: {}'.format(pretrained_dict))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logging.info('load ok')

    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('#############training############')
    n_epochs = int(config['epochs'])
    iter_num = 0
    epoch_init=0
    for epoch in range(epoch_init, n_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        # scheduler.step()
        writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        b_start = time.time()
        loader = loaders['train']
        total = len(loader)
        for batch_id, (batch_x, batch_y, patient_id) in enumerate(loader):
            iter_num = iter_num + 1
            batch_x, batch_y = batch_x.cuda(non_blocking=True), batch_y.cuda(non_blocking=True)
            # logging.info(batch_y.size())
            mask_idx = np.random.choice(15, 1)
            mask = torch.from_numpy(mask_array[mask_idx])
            batchsize = batch_x.size(0)
            mask = mask.repeat(batchsize, 1)
            mask = mask.cuda(non_blocking=True)
            model.module.is_training = True
            fuse_pred, sep_preds, prm_preds = model(batch_x, mask)
            ###Loss compute
            fuse_cross_loss = losses.softmax_weighted_loss(fuse_pred, batch_y, num_cls=config['number_classes'])
            fuse_dice_loss = losses.dice_loss(fuse_pred, batch_y, num_cls=config['number_classes'])
            fuse_loss = fuse_cross_loss + fuse_dice_loss

            sep_cross_loss = torch.zeros(1).cuda().float()
            sep_dice_loss = torch.zeros(1).cuda().float()
            for sep_pred in sep_preds:
                sep_cross_loss += losses.softmax_weighted_loss(sep_pred, batch_y, num_cls=config['number_classes'])
                sep_dice_loss += losses.dice_loss(sep_pred, batch_y, num_cls=config['number_classes'])
            sep_loss = sep_cross_loss + sep_dice_loss


            prm_cross_loss = torch.zeros(1).cuda().float()
            prm_dice_loss = torch.zeros(1).cuda().float()
            for prm_pred in prm_preds:
                prm_cross_loss += losses.softmax_weighted_loss(prm_pred, batch_y, num_cls=config['number_classes'])
                prm_dice_loss += losses.dice_loss(prm_pred, batch_y, num_cls=config['number_classes'])
            prm_loss = prm_cross_loss + prm_dice_loss

            loss = fuse_loss + sep_loss + prm_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step = (batch_id+1) + epoch*total

            writer.add_scalar('loss', loss.item(), global_step=step)
            writer.add_scalar('fuse_cross_loss', fuse_cross_loss.item(), global_step=step)
            writer.add_scalar('fuse_dice_loss', fuse_dice_loss.item(), global_step=step)
            writer.add_scalar('sep_cross_loss', sep_cross_loss.item(), global_step=step)
            writer.add_scalar('sep_dice_loss', sep_dice_loss.item(), global_step=step)
            writer.add_scalar('prm_cross_loss', prm_cross_loss.item(), global_step=step)
            writer.add_scalar('prm_dice_loss', prm_dice_loss.item(), global_step=step)

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), n_epochs, (batch_id+1), total, loss.item())
            msg += 'fusecross:{:.4f}, fusedice:{:.4f},'.format(fuse_cross_loss.item(), fuse_dice_loss.item())
            msg += 'sepcross:{:.4f}, sepdice:{:.4f},'.format(sep_cross_loss.item(), sep_dice_loss.item())
            msg += 'prmcross:{:.4f}, prmdice:{:.4f},'.format(prm_cross_loss.item(), prm_dice_loss.item())

            logging.info(msg)
        b_train = time.time()
        logging.info('train time per epoch: {}'.format(b_train - b_start))

        #########model save
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            },
            file_name)

        if (epoch+1) % 50 == 0 or (epoch>=(n_epochs-10)):
            file_name = os.path.join(ckpts, 'model_{}.pth'.format(epoch+1))
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                },
                file_name)


    msg = 'total time: {:.4f} hours'.format((time.time() - start)/3600)
    logging.info(msg)

    ##########Test the last epoch model

    # checkpoint = torch.load('/home/sjj/M2FTrans/M2FTrans_v2/output/brats2018_m2ftrans/model_last.pth')
    # pretrained_dict = checkpoint['state_dict']
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # # logging.info('pretrained_dict: {}'.format(pretrained_dict))
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    # logging.info('load ok')

    test_score = AverageMeter()
    with torch.no_grad():
        logging.info('###########test last epoch model###########')

        for i, mask in enumerate(mask_array[::-1]):
            logging.info('{}'.format(mask_name[::-1][i]))
            dice_score = test_center(
                            loaders['eval'],
                            model,
                            feature_mask = mask,
                            mask_name = mask_name[::-1][i]
                            )

            test_score.update(dice_score)
        logging.info('Avg scores: {}'.format(test_score.avg))



if __name__ == '__main__':
    main()