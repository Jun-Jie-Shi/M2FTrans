import torch
import torch.nn as nn
import torch.nn.functional as F

cross_entropy = F.cross_entropy

def dice_loss(output, target, num_cls=4, eps=1e-7):
    target = target.float()
    for i in range(num_cls):
        num = torch.sum(output[:,i,:,:,:] * target[:,i,:,:,:])
        l = torch.sum(output[:,i,:,:,:])
        r = torch.sum(target[:,i,:,:,:])
        if i == 0:
            dice = 2.0 * num / (l+r+eps)
        else:
            dice += 2.0 * num / (l+r+eps)
    return 1.0 - 1.0 * dice / num_cls

def softmax_weighted_loss(output, target, num_cls=4):
    target = target.float()
    B, _, H, W, Z = output.size()
    for i in range(num_cls):
        outputi = output[:, i, :, :, :]
        targeti = target[:, i, :, :, :]
        weighted = 1.0 - (torch.sum(targeti, (1,2,3)) * 1.0 / torch.sum(target, (1,2,3,4)))
        weighted = torch.reshape(weighted, (-1,1,1,1)).repeat(1,H,W,Z)
        if i == 0:
            cross_loss = -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
        else:
            cross_loss += -1.0 * weighted * targeti * torch.log(torch.clamp(outputi, min=0.005, max=1)).float()
    cross_loss = torch.mean(cross_loss)
    return cross_loss

