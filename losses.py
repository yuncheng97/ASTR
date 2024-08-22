import os
import torch
import numpy as np
import torch.nn.functional as F




def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def KL_loss(pred, target):
    eps     = 1e-6
    pred    = pred / pred.sum() 
    target  = target / target.sum()
    loss    = (target * torch.log(target/(pred+eps) + eps)).sum()
    return loss 

def CC_loss(pred, target):
    pred    = (pred - pred.mean()) / pred.std()  
    target  = (target - target.mean()) / target.std()
    loss    = (pred * target).sum() / (torch.sqrt((pred*pred).sum() * (target * target).sum()))
    loss    = 1 - loss
    return loss

def NSS_loss(pred, target):
    ref     = (target - target.mean()) / target.std()
    pred    = (pred - pred.mean()) / pred.std()
    loss    = (ref*target - pred*target).sum() / target.sum()
    return loss 

def bce_dice_mae_loss(pred, mask):
    bce     = F.binary_cross_entropy_with_logits(pred, mask)
    pred    = torch.sigmoid(pred)
    inter   = ((pred*mask)).sum(dim=(2,3))
    union   = ((pred+mask)).sum(dim=(2,3))
    dice    = 1-(2*inter/(union+1)).mean()
    mae     = F.l1_loss(pred, mask)
    return bce + dice + mae


def focal_loss(pred, target):
    pred = pred.squeeze()
    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = (target==1).float()
    neg_inds = (target<1).float()
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)
    
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = torch.log(pred)*torch.pow(1-pred, 2)*pos_inds
    neg_loss = torch.log(1-pred)*torch.pow(pred, 2)*neg_weights*neg_inds
    
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    loss     = -(pos_loss + neg_loss) / num_pos
    return loss

def reg_l1_loss(pred, target, mask):
    #--------------------------------#
    #   计算l1_loss
    #--------------------------------#
    pred        = pred.permute(0,2,3,1)
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,2)
    loss        = F.l1_loss(pred*expand_mask, target*expand_mask, reduction='sum')
    loss        = loss / (mask.sum() + 1e-4)
    return loss

def focal_mse_loss(pred, mask):
    focal = focal_loss(pred, mask)
    mse   = F.mse_loss(pred, mask)

    return 0.1*focal+mse

def dice_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)).sum(dim=(2,3))
    union = ((pred+mask)).sum(dim=(2,3))
    dice  = 1-(2*inter/(union+1)).mean()
    return dice