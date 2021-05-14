#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('/mindhive/nklab5/users/hlk/packages/RAFT/core/')
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from utils.frame_utils import readFlow as read_flow
from utils import flow_viz


# In[ ]:


# Read a .flo file, transform it into an image, and turn it into a tensor.
def read_flo_to_img_tensor(flo_file):
    return torch.from_numpy(flow_viz.flow_to_image(read_flow(flo_file))) / 255


# In[ ]:


class Fove8nDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_videos_dir,
                 preprocessed_tensors_dir_pattern,
                 optical_flow_frames_dir_pattern='',
                 labels_file_pattern='labels.txt'):
        self.root_videos_dir = root_videos_dir
        self.preprocessed_tensors_dir_pattern = preprocessed_tensors_dir_pattern
        self.use_optical_flow = False
        if optical_flow_frames_dir_pattern:
            self.use_optical_flow = True
            self.optical_flow_frames_dir_pattern = optical_flow_frames_dir_pattern
            # else -> None type

        # Index all the video samples and their labels.
        # Assume all video sample directories have label files.
        # List of: (video_sample_dir, pandas dataframe of labels)
        self.sample_dirs_and_labels = []
        subdir_names = [f.name for f in os.scandir(self.root_videos_dir) if f.is_dir()]
        for dir_name in subdir_names:
            sample_path = os.path.join(root_videos_dir, dir_name)
            labels_path = os.path.join(sample_path, labels_file_pattern)
            # dim: [FRAMES x 3]
            frame_labels = torch.from_numpy(np.loadtxt(labels_path))
            frames_path_and_labels = (sample_path, frame_labels)

            self.sample_dirs_and_labels.append(frames_path_and_labels)
        

    def __len__(self):
        return len(self.sample_dirs_and_labels)


    def __getitem__(self, idx):
        (video_dir_path, frame_labels) = self.sample_dirs_and_labels[idx]
        # Make sure to sort frames which have format: frame000NN
        frame_tensor_files = sorted(glob.glob(os.path.join(video_dir_path, self.preprocessed_tensors_dir_pattern, '*.pt')))
        # Each tensor has shape: [1 x CHANNELS x HEIGHT x WIDTH]
        frame_tensors = [torch.load(f) for f in frame_tensor_files]

        # Stack on the 0th dimension, from first to last.
        # dim: [FRAMES x CHANNELS x HEIGHT x WIDTH]
        frames_tensor = torch.vstack(frame_tensors)

        # Add the optical flow images as extra channels to the frames tensors.
        if self.use_optical_flow:
            flo_files = sorted(glob.glob(os.path.join(video_dir_path, self.optical_flow_frames_dir_pattern, '*.flo')))
            flo_img_tensors = [read_flo_to_img_tensor(flo_file) for flo_file in flo_files]
            # The first optical flow frame is always missing-- duplicate the second
            # frame so that dimensions match with the raw video frames.
            flo_img_tensors.insert(0, flo_img_tensors[0])
            # dim: [FRAMES x HEIGHT x WIDTH x CHANNELS]
            flo_imgs_tensor = torch.vstack([f.unsqueeze(dim=0) for f in flo_img_tensors])
            # permute to match frames tensor: [FRAMES x CHANNELS x HEIGHT x WIDTH]
            flo_imgs_tensor = flo_imgs_tensor.permute(0, 3, 1, 2)

            # Concatenate the optical flow channels onto the frames' CNN channel outputs.
            frames_tensor = torch.cat((frames_tensor, flo_imgs_tensor), dim=1)

        # Number of frames and labels (x, y, r) should be the same.
        assert frames_tensor.shape[0] == frame_labels.shape[0]

        # Resize the frames (and labels) to a square space.
        final_frames, final_labels = resize_frame_and_labels(padded_frames,
                                                             padded_labels,
                                                             self.resize_size)

        return frames_tensor.float(), frame_labels.float()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fove8n data loader test run')
    parser.add_argument('--root_videos_dir', required=True)
    parser.add_argument('--preprocessed_tensors_dir_pattern', required=True)
    parser.add_argument('--optical_flow_frames_dir_pattern', default='')
    args = parser.parse_args()

    print(args)

    train_dataset = Fove8nDataset(root_videos_dir = args.root_videos_dir,
                                  preprocessed_tensors_dir_pattern = args.preprocessed_tensors_dir_pattern,
                                  optical_flow_frames_dir_pattern = args.optical_flow_frames_dir_pattern)

    print('Batch size: 2')
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    print('Running one enumeration of the training dataloader.')
    for (idx, (input, labels)) in enumerate(train_dataloader):
        print('idx: {}\ninput.shape: {}\nlabels.shape: {}'.format(idx, input.shape, labels.shape))

