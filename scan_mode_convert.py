import os
import cv2
import math
import numpy as np

from tqdm import tqdm

def linear_to_convex(im, angle, r0=0, k=2, top=True,rotate=0):
    """
    Convert linear-array image to a convex-array (or sector-shaped) image.

    Parameters:
    im (ndarray)      - Input image
    angle (float)     - Circular sector angle in degrees
    r0 (int)          - Inner circle radius; r0=0 outputs a sector
    k (int)           - Neighboring point interpolation density (increase if white noise appears)
    top (bool)        - Center of the circle is at the top if True
    rotate (int)      - Rotation angle; positive for counter-clockwise, negative for clockwise
    """

    h, w, d = im.shape 
    
    if r0 > 0: # If output is a ring, add a white background to the top or bottom of the image
        bg = np.ones((r0, w, d), dtype=np.uint8)
        im = np.append(bg, im, axis=0) if top else np.append(im, bg, axis=0)
    
    h, w, d = im.shape 
    r = 2*h-1  
    im_fan = np.zeros((r, r, d), dtype=np.uint8) # Initialize output image array (fully transparent)
    
    idx = np.arange(h) if top else np.arange(h)[::-1]
    alpha = np.radians(np.linspace(-angle/2, angle/2, k*w)) # Generate angle sequence for sector
    for i in range(k*w): # Iterate through each column of the input image
        rows = np.int32(np.ceil(np.cos(alpha[i])*idx)) + r//2
        cols = np.int32(np.ceil(np.sin(alpha[i])*idx)) + r//2
        im_fan[(rows, cols)] = im[:,i//k]
    
    if 360 > angle > 180:# Crop out the blank area at the top of the output image
        im_fan = im_fan[int(h*(1-np.sin(np.radians((angle/2-90))))):]
    
    if not top:
        im_fan = im_fan[::-1]
    H,W,D = im_fan.shape
    # im_fan = im_fan[0:H//2,:,:]
    im_fan = np.flip(im_fan, (0,1))
    return im_fan

def convex_to_linear(img, inner_circle_rad=None, outer_circle_rad=None, x_min=None, x_max=None):
    """
    Convert a convex-array image to a linear-array image.
    
    Parameters:
    img (ndarray)           - Input circular image
    inner_circle_rad (int)  - Radius of the inner circle (default is calculated from the image)
    outer_circle_rad (int)  - Radius of the outer circle (default is the height of the image)
    x_min (int)             - Minimum x-coordinate for cropping (default is calculated from the image)
    x_max (int)             - Maximum x-coordinate for cropping (default is calculated from the image)
    
    Returns:
    img_out (ndarray)       - Output rectangular image
    inner_circle_rad (int)  - Radius of the inner circle used
    outer_circle_rad (int)  - Radius of the outer circle used
    x_min (int)             - Minimum x-coordinate used for cropping
    x_max (int)             - Maximum x-coordinate used for cropping
    """

    H,W,C            = img.shape
    inner_circle_rad = np.nonzero(img[:,W//2,0])[0][0] if inner_circle_rad is None else inner_circle_rad
    outer_circle_rad = H-1 if outer_circle_rad is None else outer_circle_rad
    radius_width     = outer_circle_rad-inner_circle_rad
    radius           = inner_circle_rad + int(outer_circle_rad - inner_circle_rad)//2
    circle_center    = (W//2,0)
    img_out          = np.zeros((radius_width,int(2*radius*math.pi),3),dtype='uint8')
    for row in range(0,img_out.shape[0]):
        for col in range(0,img_out.shape[1]):
            theta = math.pi*2/img_out.shape[1]*(col+1)
            rho   = (outer_circle_rad-row-1)
            p_x   = int(circle_center[0] + rho*math.sin(theta)+0.5)-1
            p_y   = int(circle_center[1] - rho*math.cos(theta)+0.5)-1
            if 0 <= p_x < W and 0 <= p_y < H:
                img_out[row,col,:] = img[p_y,p_x,:]
    if x_min is None:
        non_zero_index = np.nonzero(img_out[:,:,0])
        x_min = np.min(non_zero_index[1])
        x_max = np.max(non_zero_index[1])
    img_out = img_out[:,x_min:x_max,:]
    img_out = np.flip(img_out, (0,1))
    return img_out, inner_circle_rad, outer_circle_rad, x_min, x_max

    
def linear_convex_conversion():
    image_dir = 'path/to/image_folder'
    mask_dir  = 'path/to/mask_folder'
    cvt_image_dir = image_dir.replace('image', 'crop_image')+'_convert'
    cvt_mask_dir = image_dir.replace('image', 'crop_mask')+'_convert'
    if type == 'linear':
        for img in tqdm(sorted(os.listdir(image_dir))):
            if img.endswith('.jpg'):
                image = cv2.imread(image_dir+'/'+img) #H,W,C
                mask = cv2.imread(mask_dir+'/'+img.replace('.jpg', '.png'))
                H,W,C = image.shape
                image_cvt = linear_to_convex(image, angle=160, r0=120, k=20, top=False)
                mask_cvt = linear_to_convex(mask, angle=160, r0=120, k=20, top=False)
                image_cvt = cv2.resize(image_cvt, (W, H))
                mask_cvt = cv2.resize(mask_cvt, (W, H))
                cv2.imwrite(os.path.join(cvt_image_dir, img), image_cvt)
                cv2.imwrite(os.path.join(cvt_mask_dir, img.replace('.jpg', '.png')), mask_cvt)
    else:
        for img in tqdm(sorted(os.listdir(image_dir))):
            if img.endswith('.jpg'):
                image = cv2.imread(image_dir+'/'+img) #H,W,C
                mask = cv2.imread(mask_dir+'/'+img.replace('.jpg', '.png'))
                H,W,C = image.shape
                image_cvt, inner_circle_rad, outer_circle_rad, x_min, x_max = convex_to_linear(image)
                mask_cvt, *_ = convex_to_linear(mask, inner_circle_rad, outer_circle_rad, x_min, x_max)
                image_cvt = cv2.resize(image_cvt, (W, H))
                mask_cvt = cv2.resize(mask_cvt, (W, H))
                cv2.imwrite(os.path.join(cvt_image_dir, img), image_cvt)
                cv2.imwrite(os.path.join(cvt_mask_dir, img.replace('.jpg', '.png')), mask_cvt)
