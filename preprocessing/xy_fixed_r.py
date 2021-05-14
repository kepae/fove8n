#!/usr/bin/env python
# coding: utf-8

# # Make x,y,r labels for training

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.append('/mindhive/nklab5/users/hlk/packages/RAFT/core/')

from icecream import ic
import glob
import matplotlib.pyplot as plt

import numpy as np
from scipy import ndimage
from skimage import color

from utils.frame_utils import readFlow as read_flow
from utils import flow_viz

import torch, torchvision

# Return a circular mask of (True/False) 2d array
def create_circular_mask(shape, x, y, r):
    h, w = shape

    Y, X = np.ogrid[:h, :w]
    dist_from_center_mat = np.sqrt((X - x)**2 + (Y - y)**2)

    mask = dist_from_center_mat <= r
    return mask

def get_masked_img(img, x, y, r):
    mask = create_circular_mask(img.shape, x, y, r)
    masked = img.copy()
    masked[~mask] = 0
    return masked

def get_xy_coord_for_flo(flo_im, show_plot=False):
    # flo_im = flow_viz.flow_to_image(read_flow(flo_file))
    if show_plot:
        ic("visualized flow file")
        plt.imshow(flo_im)
        plt.show()

    gray_flo = color.rgb2gray(flo_im)
    if show_plot:
        ic("grayscale flow")
        plt.imshow(gray_flo, cmap='gray')
        plt.show()

    # Invert the intensity so that "active" spots have higher numerical value in range [0, 1]
    gray_flo = (gray_flo - 1 ) * -1
    if show_plot:
        ic("grayscale flow (inverted high=1)")
        plt.imshow(gray_flo, cmap='gray')
        plt.show()
        ic(ndimage.measurements.center_of_mass(gray_flo))

    # Exponeniate the pixel values; they will still remain in [0, 1]
    gray_flo = gray_flo ** 4
    if show_plot:
        ic("exponentiated (** 4) grayscale flow")
        plt.imshow(gray_flo, cmap='gray')
        plt.show()
        ic(ndimage.measurements.center_of_mass(gray_flo))

    (h, w) = ndimage.measurements.center_of_mass(gray_flo)
    return (w, h)  # return in (x, y) format


# ## Generate labels (x, y, and fixed radius)

def generate_fove8n_labels(input_output_dirs):
    '''
    input_output_dirs: list of tuple of (input frame directory,
                                         input optical flow frame directory,
                                         output label file directory)
    '''
    for inDir, outDir in input_output_dirs:
        # Gather and sort all of the frame and optical flow files.
        frame_files = glob.glob(os.path.join(inDir, '*.png'))
        frame_files = sorted(frame_files)
        flo_files = glob.glob(os.path.join(inDir, '*.flo'))
        flo_files = sorted(flo_files)

        # Generate the candidate focal points for each optical flow frame.
        xyrs = [(170,128,25)] # start fixation in the center
        for flo_file in flo_files:
            flo_im = flow_viz.flow_to_image(read_flow(flo_file))
            (x, y) = get_xy_coord_for_flo(flo_im)  # returns (height, width)
            xyrs.append((x,y,25))

        # Write the labels as text files to the given directory.
        with open(os.path.join(outDir, 'labels.txt'), 'w') as labels_file:
            for (x, y, r) in xyrs:
                label_line = "{} {} {}\n".format(x, y, r)
                labels_file.write(label_line)


# # Get the labels for each frame

# Input directories for frames and output directories for labels

tFrameDir = '/mindhive/nklab5/users/hlk/projects/vidDNN/momentsSubset/cropFrames/training/'
tLabelDir = '/mindhive/nklab5/users/hlk/projects/vidDNN/momentsSubset/labels/training/'
vFrameDir = '/mindhive/nklab5/users/hlk/projects/vidDNN/momentsSubset/cropFrames/validation/'
vLabelDir = '/mindhive/nklab5/users/hlk/projects/vidDNN/momentsSubset/labels/validation/'

### training set ###
# get tuples for the in/out directories for each label
in_out_dirs = []
cats = os.listdir(tFrameDir)
for c in cats:
    cDir = tFrameDir + c + '/'
    newLdir = tLabelDir + c + '/'
    os.mkdir(newLdir)
    vids = os.listdir(cDir)   
    for v in vids:
        vDir = cDir + v + '/'
        newerLdir = newLdir + v + '/'
        os.mkdir(newerLdir)
        in_out_dirs.append((vDir,newerLdir))

# Get the x, y, and r information for each frame
generate_fove8n_labels(in_out_dirs)

### validation set ###
# get tuples for the in/out directories for each label
in_out_dirs = []
cats = os.listdir(vFrameDir)
for c in cats:
    cDir = vFrameDir + c + '/'
    newLdir = vLabelDir + c + '/'
    os.mkdir(newLdir)
    vids = os.listdir(cDir)
    for v in vids:
        vDir = cDir + v + '/'
        newerLdir = newLdir + v + '/'
        os.mkdir(newerLdir)
        in_out_dirs.append((vDir,newerLdir))

# Get the x, y, and r information for each frame
generate_fove8n_labels(in_out_dirs)
