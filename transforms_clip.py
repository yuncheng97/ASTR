"""
Transforms and data augmentation for sequence level images, bboxes and masks.
"""
import cv2
import PIL
import torch
import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from numpy import random as rand

def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    assert mode in ['iou', 'iof']
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start, 0) * np.maximum(y_end - y_start, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        union = np.maximum(union, eps)
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def crop(clip, target, region):
    cropped_image = []
    for image in clip:
        cropped_image.append(F.crop(image, *region))

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    return cropped_image, target



def pad(clip, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = []
    for image in clip:
        padded_image.append(F.pad(image, (0, 0, padding[0], padding[1])))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[0].size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img[0].width, self.max_size))
        h = random.randint(self.min_size, min(img[0].height, self.max_size))
        region = T.RandomCrop.get_params(img[0], [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class MinIoURandomCrop(object):
    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        self.min_ious = min_ious
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, img, target):
        w,h = img.size
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return img,target
            min_iou = mode
            boxes = target['boxes'].numpy()
            labels = target['labels']

            for i in range(50):
                new_w = rand.uniform(self.min_crop_size * w, w)
                new_h = rand.uniform(self.min_crop_size * h, h)
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue
                left = rand.uniform(w - new_w)
                top = rand.uniform(h - new_h)
                patch = np.array((int(left), int(top), int(left + new_w), int(top + new_h)))
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue
                
                if len(overlaps) > 0:
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) * (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                        return mask
                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if False in mask:
                        continue
                    #TODO: use no center boxes
                    #if not mask.any():
                    #    continue

                    boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                    boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                    boxes -= np.tile(patch[:2], 2)
                    target['boxes'] = torch.tensor(boxes)
                
                img = np.asarray(img)[patch[1]:patch[3], patch[0]:patch[2]]
                img = Image.fromarray(img)
                width, height = img.size
                target['orig_size'] = torch.tensor([height,width])
                target['size'] = torch.tensor([height,width])
                return img,target 


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
    def __call__(self, image, target):
        
        if rand.randint(2):
            alpha = rand.uniform(self.lower, self.upper)
            image *= alpha
        return image, target

class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta
    def __call__(self, image, target):
        if rand.randint(2):
            delta = rand.uniform(-self.delta, self.delta)
            image += delta
        return image, target

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, target):
        if rand.randint(2):
            image[:, :, 1] *= rand.uniform(self.lower, self.upper)
        return image, target

class RandomHue(object): #
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, target):
        if rand.randint(2):
            image[:, :, 0] += rand.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, image, target):
        if rand.randint(2):
            swap = self.perms[rand.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, target

class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, target):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, target

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, image):
        image = image[:, :, self.swaps]
        return image

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
    
    def __call__(self,clip,target):
        imgs = []
        for img in clip:
            img = np.asarray(img).astype('float32')
            img, target = self.rand_brightness(img, target)
            if rand.randint(2):
                distort = Compose(self.pd[:-1])
            else:
                distort = Compose(self.pd[1:])
            img, target = distort(img, target)
            img, target = self.rand_light_noise(img, target)
            imgs.append(Image.fromarray(img.astype('uint8')))
        return imgs, target

#NOTICE: if used for mask, need to change
class Expand(object):
    def __init__(self, mean):
        self.mean = mean
    def __call__(self, clip, target):
        if rand.randint(2):
            return clip,target
        imgs = []
        masks = []
        image = np.asarray(clip[0]).astype('float32')
        height, width, depth = image.shape
        ratio = rand.uniform(1, 4)
        left = rand.uniform(0, width*ratio - width)
        top = rand.uniform(0, height*ratio - height)
        for i in range(len(clip)):
            image = np.asarray(clip[i]).astype('float32')
            expand_image = np.zeros((int(height*ratio), int(width*ratio), depth),dtype=image.dtype)
            expand_image[:, :, :] = self.mean
            expand_image[int(top):int(top + height),int(left):int(left + width)] = image
            imgs.append(Image.fromarray(expand_image.astype('uint8')))
            expand_mask = torch.zeros((int(height*ratio), int(width*ratio)),dtype=torch.uint8)
            expand_mask[int(top):int(top + height),int(left):int(left + width)] = target['masks'][i]
            masks.append(expand_mask)
        boxes = target['boxes'].numpy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        target['boxes'] = torch.tensor(boxes)
        target['masks']=torch.stack(masks)
        return imgs, target






class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)





class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, targets=None, maps=None):
        normalized_images  = []

        for image in images:
            normalized_images.append(F.normalize(image, mean=self.mean, std=self.std))
        if targets is None:
            return normalized_images, None, None

        return normalized_images, targets, maps

class ToTensor(object):
    def __call__(self, images, targets, maps):
        tensor_images  = []
        tensor_targets = []
        tensor_maps    = []
        for (image, target, map) in zip(images, targets, maps):
            tensor_images.append(F.to_tensor(image))
            tensor_targets.append(F.to_tensor(target))
            tensor_maps.append(F.to_tensor(map))

        return tensor_images, tensor_targets, tensor_maps

def resize(images, targets, maps, size):
    rescaled_images = []
    rescaled_targets = []
    rescaled_maps = []
    for image, target, map in zip(images, targets, maps):
        rescaled_images.append(F.resize(image, (size,size)))
        rescaled_targets.append(F.resize(target, (size,size)))
        rescaled_maps.append(F.resize(map, (size,size)))

    return rescaled_images, rescaled_targets, rescaled_maps

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, images, targets, maps):
        images, targets, maps = resize(images, targets, maps, self.size)
        return images, targets, maps

def hflip(images, targets, maps):
    flipped_images  = []
    flipped_targets = []
    flipped_maps    = []

    for (image, target, map) in zip(images, targets, maps):
        flipped_images.append(F.hflip(image))
        flipped_targets.append(F.hflip(target))
        flipped_maps.append(F.hflip(map))
    
    return flipped_images, flipped_targets, flipped_maps

def vflip(images, targets, maps):

    flipped_images  = []
    flipped_targets = []
    flipped_maps    = []

    for (image, target, map) in zip(images, targets, maps):
        flipped_images.append(F.vflip(image))
        flipped_targets.append(F.vflip(target))
        flipped_maps.append(F.vflip(map))
    
    return flipped_images, flipped_targets, flipped_maps

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images, targets, maps):
        if random.random() < self.p:
            return vflip(images, targets, maps)
        return images, targets, maps

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, images, targets, maps):
        if random.random() < self.p:
            return hflip(images, targets, maps)
        return images, targets, maps

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, targets, maps):
        for t in self.transforms:
            images, targets, maps = t(images, targets, maps)
        return images, targets, maps

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string