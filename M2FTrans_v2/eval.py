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
        format='%(asctime)s %(message)s', filename='./logs/eval_brats2018_m2ftrans.txt')
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
            exit(0)


if __name__ == '__main__':
    main()