#!/usr/bin/env python
# coding: utf-8

# # Optic Flow 

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('/mindhive/nklab5/users/hlk/packages/RAFT/core/')
import argparse
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
from argparse import Namespace
from utils import flow_viz
from utils import frame_utils
DEVICE = 'cuda' 

# Extract Optic flow for images using pre-trained model

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def get_optic_flow(vidPath,mPath):
    # set the parameters for the model
    args = Namespace(alternate_corr=False, mixed_precision=False, model=mPath, path=vidPath, small=False)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()
    
    # use the pre-trained model
    with torch.no_grad():
        # get all of the frame names
        images = glob.glob(os.path.join(args.path, '*.png')) +                      glob.glob(os.path.join(args.path, '*.jpg'))
        images = sorted(images)
        frame = 1
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)
            
            # pad the images
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            
            # extract the optic flow for each image
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        
            # save the optic flow file for each frame
            frame = frame + 1
            flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
            output_file = os.path.join(args.path, 'of%04d.flo' % (frame))
            frame_utils.writeFlow(output_file, flo)

# Get the optic flow for each frame in each movie and save the output

# optic flow model
mPath = '/mindhive/nklab5/users/hlk/projects/vidDNN/fove8n/learn/RAFT/models/raft-things.pth'

# optic flow for training images
tDir = '/mindhive/nklab5/users/hlk/projects/vidDNN/momentsSubset/cropFrames/training/'
cats = glob.glob(os.path.join(tDir, '*'))
for cDir in cats:
    vids = glob.glob(os.path.join(cDir,'*'))   
    for v in vids:
        vidPath = v + '/'
        get_optic_flow(vidPath,mPath)

# optic flow for validation images        
vDir = '/mindhive/nklab5/users/hlk/projects/vidDNN/momentsSubset/cropFrames/validation/'      
cats = glob.glob(os.path.join(vDir, '*'))
for cDir in cats:
    vids = glob.glob(os.path.join(cDir,'*'))   
    for v in vids:
        vidPath = v + '/'
        print(vidPath)
        get_optic_flow(vidPath,mPath)        
