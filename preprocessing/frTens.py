#!/usr/bin/env python
# coding: utf-8

# # Get the tensors for each frame from all videos

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
import numpy as np
import glob
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image


## Get the pre-trained model

rs50 = torchvision.models.resnet50(pretrained=True)

## Keep only the first layers

# keep the first layers of net
modules=list(rs50.children())[:4]
convNet = nn.Sequential(*modules)

# fix the parameters
for p in convNet.parameters():
    p.requires_grad=False

# keep just the first layer 
mods=list(rs50.children())[:1]
conv = nn.Sequential(*mods)

# fix the parameters
for p in conv.parameters():
    p.requires_grad=False


## Get Image Features
# The output is a 64 channel tensor ( x W x H) that can be sent to our network

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def image_loader(imgName):
    im = Image.open(imgName)
    im = transform(im)
    im = torch.unsqueeze(im,0)
    return im#.cuda()

# get a tensor for each frame and save it
fDir = '/mindhive/nklab5/users/hlk/projects/vidDNN/momentsSubset/cropFrames/training/'
tDir = '/mindhive/nklab5/users/hlk/projects/vidDNN/momentsSubset/cropTensors/training/'
cats = ['clapping','constructing'] # only 2 of 17 labels
for c in cats:
    cDir = fDir + c + '/'
    newTdir = tDir + c + '/'
    os.mkdir(newTdir)
    vids = os.listdir(cDir)   
    for v in vids:
        vDir = cDir + v + '/'
        newerTdir = newTdir + v + '/'
        os.mkdir(newerTdir)
        fs = glob.glob(os.path.join(vDir,'*jpg'))
        for f in fs:
            im = image_loader(f)
            tens = conv(im)
            tensPath = newerTdir + f[-9:-4] + '.pt' 
            torch.save(tens,tensPath)

# get a tensor for each frame and save it
fDir = '/mindhive/nklab5/users/hlk/projects/vidDNN/momentsSubset/cropFrames/validation/'
tDir = '/mindhive/nklab5/users/hlk/projects/vidDNN/momentsSubset/cropTensors/validation/'
for c in cats:
    cDir = fDir + c + '/'
    newTdir = tDir + c + '/'
    os.mkdir(newTdir)
    vids = os.listdir(cDir)
    for v in vids:
        vDir = cDir + v + '/'
        newerTdir = newTdir + v + '/'
        os.mkdir(newerTdir)
        fs = glob.glob(os.path.join(vDir,'*jpg'))
        for f in fs:
            im = image_loader(f)
            tens = conv(im)
            tensPath = newerTdir + f[-9:-4] + '.pt'
            torch.save(tens,tensPath)

