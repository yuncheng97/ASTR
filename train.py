import os
import sys
import cv2
import time
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from tabulate import tabulate
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
from torch.cuda.amp import autocast, GradScaler
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader, DistributedSampler

from model import SegModel, ASTR
import segmentation_models_pytorch as smp
from dataset import TrainDataset, TestDataset
from utils import clip_gradient, adjust_lr, ModelEma, init_distributed_mode, get_rank
from losses import bce_dice_mae_loss


def train(args, train_loader, val_loader, model, optimizer, epoch, total_step, save_path, ema, scalar, writer):
    global step
    model.train()
    loss_all   = 0
    epoch_step = 0
    for i, (images, targets, bodygt) in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, targets, bodygt = images.cuda(), targets.cuda(), bodygt[:,1:].cuda()
        
        pred, bodymap = model(images)
        
        pred     = F.interpolate(pred, size=targets.shape[3:], mode='bilinear', align_corners=True)
        bodymap  = F.interpolate(bodymap.view(-1,1,bodymap.shape[3],bodymap.shape[4]), size=targets.shape[3:], mode='bilinear', align_corners=True)
        bodygt   = bodygt.reshape(-1, 1, bodygt.shape[3], bodygt.shape[4])

        if scalar:
            with autocast():
                seg_loss  = bce_dice_mae_loss(pred, targets[:,0])
                body_loss = bce_dice_mae_loss(bodymap, bodygt)
                theta     = 0.3
                all_loss  = seg_loss+ theta*body_loss

            scaler.scale(all_loss).backward()
            scaler.step(optimizer)
            scaler.update() 

        else:
            seg_loss  = bce_dice_mae_loss(pred, targets[:,0])
            body_loss = bce_dice_mae_loss(bodymap, bodygt)
            theta     = 0.3
            all_loss  = seg_loss+ theta*body_loss

            all_loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()

        if ema:
            ema.update(model) 

        step       += 1
        epoch_step += 1

        loss_all   += all_loss.data
        if i % 50 == 0 or i == len(train_loader) or i==1:
            print(f'TRAIN: Epoch [{epoch:03d}/{args.epoch:03d}], Step [{step:04d}/{total_step:04d}], Lr: {optimizer.param_groups[0]["lr"]}, Loss: {all_loss.data:.4f}, seg loss: {seg_loss.data:.4f}, body loss: {body_loss.data:.4f}')
            logging.info(f'TRAIN: Epoch [{epoch:03d}/{args.epoch:03d}], Step [{step:04d}/{total_step:04d}], Lr: {optimizer.param_groups[0]["lr"]}, Loss: {all_loss.data:.4f}, seg loss: {seg_loss.data:.4f}, body loss: {body_loss.data:.4f}')

        if step >=500 and step % 100 == 0:
            val(val_loader, model, step, epoch, exp_path, writer)
        writer.add_scalar('Loss-iteration', all_loss.data, global_step=step)

    loss_all/=epoch_step
    logging.info(f'TRAIN: Epoch [{epoch:03d}/{args.epoch:03d}], Loss_AVG: {loss_all:.4f}')
    writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)


#test function
def val(test_loader, model, step, epoch, save_path, writer):
    global best_mae, best_dice, best_acc, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum, iou_sum, dice_sum, sen_sum, spe_sum, acc_sum, seconds  = 0, 0, 0, 0, 0, 0, 0
        for i in tqdm(range(test_loader.size)):
            image, target, *_ = test_loader.load_data()
            target    = np.asarray(target, np.float32)
            target    /= (target.max() + 1e-8)
            image     = image.cuda()

            start     = time.time()
            pred,_    = model(image)
            end       = time.time()
            seconds += end - start

            pred      = F.interpolate(pred, size=target.shape, mode='bilinear', align_corners=False)
            res       = pred.clone()
            res       = res.sigmoid().data.cpu().numpy().squeeze()
            res       = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum   += np.sum(np.abs(res-target))*1.0/(target.shape[0]*target.shape[1])

            bi_pred       = torch.from_numpy((pred[0,0] > 0).cpu().numpy())
            tp, fp, fn, tn = smp.metrics.get_stats(bi_pred.unsqueeze(0).unsqueeze(0), torch.from_numpy(target).unsqueeze(0).unsqueeze(0).long(), mode='binary', threshold=0.5)
            iou_sum   += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            dice_sum  += smp.metrics.dice_score(tp, fp, fn, tn, reduction="micro")
            sen_sum   += smp.metrics.sensitivity(tp, fp, fn, tn, reduction="micro")
            spe_sum   += smp.metrics.specificity(tp, fp, fn, tn, reduction="micro")
            acc_sum   += smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

            # inter     = (bi_pred * torch.from_numpy(target)).sum(dim=(0,1))
            # union     = bi_pred.sum(dim=(0,1)) + torch.from_numpy(target).sum(dim=(0,1))
        fps     = test_loader.size / seconds
        mae     = mae_sum/test_loader.size
        iou     = iou_sum/test_loader.size
        dice    = dice_sum/test_loader.size
        sen     = sen_sum/test_loader.size
        spe     = spe_sum/test_loader.size
        acc     = acc_sum/test_loader.size

        writer.add_scalar('MAE', mae, global_step=step)
        writer.add_scalar('I0U', iou, global_step=step)
        writer.add_scalar('Dice', dice, global_step=step)
        writer.add_scalar('Sen', sen, global_step=step)
        writer.add_scalar('Spe', spe, global_step=step)
        writer.add_scalar('Acc', acc, global_step=step)

        if mae < best_mae:
            best_mae   = mae
            best_epoch = epoch
            torch.save(model.state_dict(), save_path+'/epoch_bestMAE.pth')
            print(f'best MAE {best_mae:.3f} iteration:{step} / epoch:{epoch}')
            logging.info(f'best MAE {best_mae:.3f} iteration:{step} / epoch:{epoch}')
            
        if dice > best_dice:
            best_dice   = dice
            torch.save(model.state_dict(), save_path+'/epoch_bestDice.pth')
            print(f'best Dice {best_dice:.3f} (IOU: {iou:.3f}) iteration:{step} / epoch:{epoch}')
            logging.info(f'best Dice {best_dice:.3f} (IOU: {iou:.3f}) iteration:{step} / epoch:{epoch}')
                
        print(f'TEST: Iteration: {step} MAE: {mae:.3f} IoU: {iou:.3f} Dice: {dice:.3f} Sen: {sen:.3f} Spe: {spe:.3f} Acc: {acc:.3f} fps: {fps:.3f} ####  bestMAE: {best_mae:.3f} bestDice: {best_dice:.3f}')
        logging.info(f'TEST: Iteration: {step} MAE: {mae:.3f} IoU: {iou:.3f} Dice: {dice:.3f} Sen: {sen:.3f} Spe: {spe:.3f} Acc: {acc:.3f} fps: {fps:.3f} ####  bestMAE: {best_mae:.3f} bestDice: {best_dice:.3f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',       type=int,   default=20,                        help='epoch number'                         )             
    parser.add_argument('--lr',          type=float, default=1e-4,                      help='learning rate'                        )
    parser.add_argument('--backbone',    type=str,   default='res2net50',               help='model backbone'                       )
    parser.add_argument('--pretrained',  type=str,   default=None,                      help='pretrained model path'                )
    parser.add_argument('--resume',      type=str,   default=None,                      help='resume model checkpoint path'         )
    parser.add_argument('--batchsize',   type=int,   default=16,                        help='training batch size'                  )
    parser.add_argument('--clip_size',   type=int,   default=3,                         help='a clip size'                          )
    parser.add_argument('--train_size',  type=int,   default=352,                       help='training dataset size'                )
    parser.add_argument('--clip',        type=float, default=0.5,                       help='gradient clipping margin'             )
    parser.add_argument('--decay_rate',  type=float, default=0.1,                       help='decay rate of learning rate'          )
    parser.add_argument('--decay_epoch', type=int,   default=60,                        help='every n epochs decay learning rate'   )
    parser.add_argument('--optimizer',   type=str,   default='adam',                    help='optimizer for training'               )
    parser.add_argument('--scheduler',   type=str,   default='cos',                     help='scheduler for training'               )
    parser.add_argument('--gpu_id',      type=str,   default='0',                       help='train use gpu'                        )
    parser.add_argument('--data_root',   type=str,   default=None,                      help='the training images root'             )
    parser.add_argument('--save_path',   type=str,   default='./results/',              help='the path to save models and logs'     )
    parser.add_argument('--note',        type=str,   default=None,                      help='the experiment note'                  )
    parser.add_argument('--augmentation',action='store_true',                           help='whether use dataset augmentation'     )
    parser.add_argument('--mp_train',    action='store_true',                           help='whether use mix precision training'   )
    parser.add_argument('--model_ema',   action='store_true',                           help='whether use model ema training'       )
    parser.add_argument('--distributed', action='store_true',                           help='use distribution training'            )
    parser.add_argument('--world_size',  type=int,   default=1,                         help='number of distributed processes'      )
    parser.add_argument('--dist_url',    default='env://',                              help='url to set up distributed training'   )
    parser.add_argument('--seed',        default=42, type=int,                          help='random seed'                          )

    args = parser.parse_args()
    os.system('')
    #set the device for training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('set ramdom seed: ', seed)
    
    if args.distributed:
        print("Using", torch.cuda.device_count(), "GPUs!")
        init_distributed_mode(args)

    
    cudnn.benchmark = True

    model = ASTR(args)

    model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    if args.model_ema:
        ema  = ModelEma(model, decay=0.9998)
    else:
        ema  = None
    if args.mp_train:
        scaler = GradScaler()
    else:
        scaler = None


    if args.resume:
        print('loading model...')
        load_path = args.resume + '/epoch_bestDice.pth'
        checkpoint = torch.load(load_path, map_location='cuda:0')
        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
        print('load model from ', load_path)

    ## optimizer ##
    optim_dict  = {
        'adam': torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4),
        'adamw':torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4),
        'sgd':  torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4),
    }
    optimizer   = optim_dict[args.optimizer]

    ## scheduler ##
    scheduler_dict = {
        'step': torch.optim.lr_scheduler.MultiStepLR(optimizer, [20], gamma=0.1, last_epoch=-1, verbose=False),
        'cos':  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6),
        'exp':  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5, last_epoch=-1, verbose=False),
    }
    scheduler      = scheduler_dict[args.scheduler]


    warmup_epochs  = args.epoch // 8
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler)
    scheduler.step()
    ## experiment path ##
    save_path          = os.path.join(args.save_path, args.note)
    current_timestamp  = datetime.now().timestamp()
    current_datetime   = datetime.fromtimestamp(current_timestamp+29220)  # different time zone
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    exp_path           = os.path.join(save_path, 'log_'+formatted_datetime)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(exp_path, exist_ok=True)

    ## dataset ##
    print('Loading data...')
    train_set = TrainDataset(args.data_root, args.train_size, args.clip_size, args.augmentation)
    if args.distributed:
        train_sampler = DistributedSampler(train_set)
        train_loader = DataLoader(dataset=train_set, batch_size=args.batchsize, shuffle=False, num_workers=6, sampler=train_sampler, pin_memory=True)
    else:
        train_loader = DataLoader(dataset=train_set, batch_size=args.batchsize, shuffle=True, num_workers=6, pin_memory=True)

    val_loader   = TestDataset(args.data_root, args.train_size, args.clip_size)    
    total_step   = len(train_loader)*int(args.epoch)


    logging.basicConfig(filename=exp_path+'/log.log',format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    tables  = [[args.backbone, train_set.__len__(), val_loader.__len__(), args.epoch, args.lr, args.batchsize, args.train_size, args.clip_size, args.scheduler, args.optimizer, args.model_ema, args.mp_train, args.distributed, exp_path, str(torch.cuda.device_count())+'*'+str(torch.cuda.get_device_name(0)), seed]]
    headers = ['backbone', 'train samples', 'val samples', 'epoch', 'lr', 'batch size', 'image size', 'clip size', 'scheduler', 'optimizer', 'ema', 'mix precision', 'distributed','save path', 'GPU', 'seed']
    print('===training configures===')
    print(tabulate(tables, headers, tablefmt="grid", numalign="center"))
    logging.info('\n'+tabulate(tables, headers, tablefmt="github", numalign="center"))


    #set loss function
    writer     = SummaryWriter(exp_path+'/summary')
    step       = 0
    best_epoch = 0
    best_mae, best_dice, best_acc, best_f1   = 1, 0, 0, 0
    
    print("===Start train===")
    logging.info("===Start train===")
    start_time = time.time()
    for epoch in range(1, args.epoch):
        train(args, 
              train_loader, 
              val_loader, 
              model, 
              optimizer, 
              epoch, 
              total_step, 
              exp_path, 
              ema, 
              scaler, 
              writer)
        scheduler.step()
        print("----"*30)
        logging.info("----"*30)

    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    print(f'Training completed.\nTotal training time: {format(total_time_str)}')
    logging.info(f'Training completed.\nTotal training time: {format(total_time_str)}')