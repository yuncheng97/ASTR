import os
import cv2
import torch
import random
import numpy as np
import transforms_clip
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image

# dataset for training
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.


class TrainDataset(data.Dataset):
    def __init__(self, root, trainsize, clip_len=3, augment=False):
        self.trainsize  = trainsize
        self.clip_len   = clip_len
        image_root      = os.path.join(root, "train/image")
        if augment:
            image_aug_root = os.path.join(root, "train_aug/image")


        self.images = []
        self.gts    = []
        # collect video clips in train set
        for video in os.listdir(image_root):
            frms = sorted(os.listdir(os.path.join(image_root, video)))
            for idx in range(len(frms)):
                clip = []
                for ii in range(clip_len):
                    pick_idx = idx + ii if idx - ii < 0 else idx - ii
                    if pick_idx >= len(frms):
                        pick_idx = - 1
                    clip.append(os.path.join(image_root, video, frms[pick_idx]))
                self.images.append(clip)
                self.gts.append([x.replace("image", "mask") for x in clip])   #cuv

        if augment:
            for video in os.listdir(image_aug_root):
                frms = sorted(os.listdir(os.path.join(image_aug_root, video)))
                for idx in range(len(frms)):
                    clip = []
                    for ii in range(clip_len):
                        pick_idx = idx + ii if idx - ii < 0 else idx - ii
                        if pick_idx >= len(frms):
                            pick_idx = - 1
                        clip.append(os.path.join(image_aug_root, video, frms[pick_idx]))
                    self.images.append(clip)
                    self.gts.append([x.replace("image", "mask") for x in clip])   #cuv

        self.size = len(self.images)
        self.transform = transforms_clip.Compose([
            transforms_clip.RandomVerticalFlip(),
            transforms_clip.RandomHorizontalFlip(),
            transforms_clip.Resize(self.trainsize),
            transforms_clip.ToTensor(),
            transforms_clip.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):
        images = [self.rgb_loader(x) for x in self.images[index]]
        gts = [self.binary_loader(x) for x in self.gts[index]]

        bodygts  = []
        for gt in gts:
            gt = np.asarray(gt)
            kernel = np.ones((25,25), np.uint8)
            bodygt = cv2.dilate(gt, kernel, 2)
            bodygt = bodygt/np.max(bodygt) if np.max(bodygt) > 0 else bodygt
            bodygt = Image.fromarray(bodygt)
            bodygts.append(bodygt)

        image, gt, bodygt = self.transform(images, gts, bodygts)
        return torch.stack(image), torch.stack(gt), torch.stack(bodygt)

    def draw_gaussian(self, heatmap, center, radius, k=1):
        diameter        = 2*radius + 1
        gaussian        = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y            = int(center[0]), int(center[1])
        height, width   = heatmap.shape[0:2]
        left, right     = min(x, radius), min(width-x, radius+1)
        top, bottom     = min(y, radius), min(height-y, radius+1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def gaussian_radius(self, height, width , min_overlap=0.7):
        a1  = 1
        b1  = (height+width)
        c1  = width*height*(1-min_overlap) / (1+min_overlap)
        sq1 = np.sqrt(b1**2 - 4*a1*c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2*(height+width)
        c2  = (1-min_overlap) * width * height
        sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
        r2  = (b2 + sq2)/2

        a3  = 4*min_overlap
        b3  = -2*min_overlap*(height + width)
        c3  = (min_overlap - 1)*width*height
        sq3 = np.sqrt(b3**2 - 4*a3*c3)
        r3  = (b3 + sq3) / 2
        return int(min(r1, r2, r3))


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size



class TestDataset(data.Dataset):
    def __init__(self, root, testsize, clip_len=3):
        self.testsize   = testsize
        image_root      = os.path.join(root, "test/image")
        
        self.images = []
        self.gts    = []
        # collect video clips in train set
        for video in os.listdir(image_root):
            frms = sorted(os.listdir(os.path.join(image_root, video)))
            for idx in range(len(frms)):
                clip = []
                for ii in range(clip_len):
                    pick_idx = idx + ii if idx - ii < 0 else idx - ii
                    if pick_idx >= len(frms):
                        pick_idx = - 1
                    clip.append(os.path.join(image_root, video, frms[pick_idx]))
                self.images.append(clip)
                self.gts.append([x.replace("image", "mask") for x in clip])   #cuv

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()

        self.size   = len(self.images)
        self.index  = 0

    def load_data(self):
        images  = [self.rgb_loader(x) for x in self.images[self.index]]
        images  = [self.transform(x) for x in images]
        gt      = cv2.imread(self.gts[self.index][0], cv2.IMREAD_GRAYSCALE)

        name    = self.images[self.index][0]
        self.index += 1
        self.index = self.index % self.size
        return torch.stack(images).unsqueeze(0), gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size
