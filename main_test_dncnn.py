import os.path
import argparse

import cv2 as cv
import numpy as np
from collections import OrderedDict

import torch
from models.network_dncnn import DnCNN as net
import torch.nn as nn
import models.basicblock as B


# Model definition

def tensor2uint16(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint16((img*65535.0).round())


def main():

    input_path = 'outputs/output_2/output_chest.png'
    out_dir = 'outputs/output_4'
    model_path = 'model_zoo/dncnn_25.pth'
    n_channels = 1        # fixed for grayscale image
    nb = 17               # fixed
    noise_level_img = 15
    need_degradation = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    print ('Model path: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    print ('Params number: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    img_L = cv.imread(input_path, -1)
    img_L = cv.resize(img_L, (800,800))
    img_L = np.expand_dims(img_L, axis=2)  # HxWx1
    
    # Convert uint16 to single
    img_L =  np.float32(img_L/65535.)

    print (type(img_L), img_L.dtype)

    if need_degradation:  # degradation process
        np.random.seed(seed=0)  # for reproducibility
        img_L += np.random.normal(0, noise_level_img/255., img_L.shape)

    # Convert single to tensor
    img_L = torch.from_numpy(np.ascontiguousarray(img_L)).permute(2, 0, 1).float().unsqueeze(0)
    
    # Prepare tensor for GPU / CPU
    img_L = img_L.to(device)

    # Inference
    img_E = model(img_L)
    img_E = tensor2uint16(img_E)
    print (img_E.dtype)

    # Saving output
    img_E = np.squeeze(img_E)
    if img_E.ndim == 3:
        img_E = img_E[:, :, [2, 1, 0]]

    # Path for output
    out_file = os.path.join(out_dir, f'denoise_{os.path.splitext(os.path.split(input_path)[-1])[0]}.png')
    cv.imwrite(out_file, img_E)


if __name__ == '__main__':
    main()
