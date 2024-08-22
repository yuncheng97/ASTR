import os
import cv2
import time
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from model import ASTR
from dataset import TestDataset
from utils import init_distributed_mode
import segmentation_models_pytorch as smp

def Quantitative(test_loader, model, save_path):
    save_txt_path = os.path.join(save_path, 'quantitative.txt')
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
        fps     = test_loader.size / seconds
        mae     = mae_sum/test_loader.size
        iou     = iou_sum/test_loader.size
        dice    = dice_sum/test_loader.size
        sen     = sen_sum/test_loader.size
        spe     = spe_sum/test_loader.size
        acc     = acc_sum/test_loader.size
                
        print(f'TEST: MAE: {mae:.3f} IoU: {iou:.3f} Dice: {dice:.3f} Sen: {sen:.3f} Spe: {spe:.3f} Acc: {acc:.3f} fps: {fps:.3f}')
        with open(save_txt_path, 'w') as f:
            line = f'MAE: {mae:.3f} IoU: {iou:.3f} Dice: {dice:.3f} Sen: {sen:.3f} Spe: {spe:.3f} Acc: {acc:.3f} fps: {fps:.3f}'
            f.write(line)

#test function
def Qualitative(test_loader, model, save_path):
    save_path = os.path.join(save_path, 'figures')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(test_loader.size)):
            image, target, name  = test_loader.load_data()
            clip_name = name.split("/")[-2] # e.g. '1'
            img_name  = name.split("/")[-1] # e.g. '001.png'
            image     = image.cuda()
            pred,_    = model(image)
            pred      = F.interpolate(pred, size=target.shape, mode='bilinear', align_corners=False)

            pred       = (pred[0,0] > 0)
            pred       = pred.cpu().numpy()
            pred       = np.expand_dims(pred,2)

            # save
            save_video_path = os.path.join(save_path, clip_name)
            if not os.path.exists(save_video_path):
                os.makedirs(save_video_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_video_path, img_name), np.uint8((pred)*255))
    return
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',    type=str,   default='res2net50',               help='model backbone'                       )
    parser.add_argument('--pretrained',  type=str,   default=None,                      help='pretrained model path'                )
    parser.add_argument('--resume',      type=str,   default=None,                      help='resume path'                          )
    parser.add_argument('--clip_size',   type=int,   default=4,                         help='a clip size'                          )
    parser.add_argument('--train_size',  type=int,   default=352,                       help='training dataset size'                )
    parser.add_argument('--gpu_id',      type=str,   default='0',                       help='train use gpu'                        )
    parser.add_argument('--data_root',   type=str,   default=None,                      help='the training images root'             )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cudnn.benchmark = True

    model = ASTR(args)
    model.cuda()

    if args.resume:
        print('loading model...')
        load_path = args.resume + '/epoch_bestDice.pth'
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint.items()})
        print('load model from ', load_path)

    print('load data...')
    test_loader  = TestDataset(args.data_root, args.train_size, args.clip_size)

    output_path = args.resume

    print("Start eval...")    
    # Quantitative(test_loader, model, output_path)
    Qualitative(test_loader, model, output_path)