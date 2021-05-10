import os
import glob

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

# Pad the frame as needed to make a square, and translate the label's (x, y, _)
# coordinates as necessary.
def square_pad_frame_and_labels(frames, labels):
    F, C, H, W = frames.shape
    max_wh = np.max([H, W])
    # Horizontal and vertical padding for each side.
    hp = int((max_wh - W) / 2)
    vp = int((max_wh - H) / 2)
    padding = (hp, vp, hp, vp)

    padded_frames = torchvision.transforms.functional.pad(frames, padding,
                                                          0, 'constant')
    
    # Extract column vectors from labels.
    x_col = labels[:, 0].reshape(-1, 1)
    y_col = labels[:, 1].reshape(-1, 1)
    r_col = labels[:, 2].reshape(-1, 1)

    # Add one side of padding to each coordinate, representing the displacement.
    x_col = x_col + hp
    y_col = y_col + vp

    # Re-assemble the tensor from the column vectors.
    padded_labels = torch.cat((x_col, y_col, r_col), dim=1)

    padded_dim_diff = abs(padded_frames.shape[2] - padded_frames.shape[3])
    assert padded_dim_diff == 0 or padded_dim_diff == 1, \
        "When square padding the frames, incorrectly generated shape: {}".format(padded_frames.shape)

    return (padded_frames, padded_labels)

# Scale the image proportionally until one side is the given size.
# (For a square input, this will product a square image!)
# Also scale the label's (x, y) coordinates as well.
def resize_frame_and_labels(frames, labels, size):
    F, C, H, W = frames.shape
    # labels dim: F x 3, representing (x, y, r) labels

    # Create a Resize transform object and apply it to the frames tensor.
    trans = transforms.Resize(size)
    resized_frames = trans(frames)

    _, _, h_new, w_new = resized_frames.shape

    h_factor = h_new / H
    w_factor = w_new / W

    # Extract column vectors from labels.
    x_col = labels[:, 0].reshape(-1, 1)
    y_col = labels[:, 1].reshape(-1, 1)
    r_col = labels[:, 2].reshape(-1, 1)

    # Scale the x and y by the new width and height factors.
    x_col = x_col * w_factor
    y_col = y_col * h_factor

    # Re-assemble the tensor from the column vectors.
    resized_labels = torch.cat((x_col, y_col, r_col), dim=1)

    return (resized_frames, resized_labels)
    

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_videos_dir,
                 preprocessed_tensors_dir_pattern,
                 labels_scale,  # to compensate for the half-size output of the convolved frames
                 optical_flow_frames_dir_pattern='',
                 labels_file_pattern='labels.txt',
                 resize_size=224):
        self.root_videos_dir = root_videos_dir
        self.preprocessed_tensors_dir_pattern = preprocessed_tensors_dir_pattern
        self.labels_scale = labels_scale
        self.resize_size = resize_size
        # self.use_optical_flow = False
        # if optical_flow_frames_dir_pattern:
        #     self.use_optical_flow = True
        #     self.optical_flow_frames_dir_pattern = optical_flow_frames_dir_pattern
        #     # else -> None type

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
        frame_tensor_files = sorted(glob.glob(\
            os.path.join(video_dir_path, self.preprocessed_tensors_dir_pattern, '*.pt')))
        # Each tensor has shape: [1 x CHANNELS x HEIGHT x WIDTH]
        frame_tensors = [torch.load(f) for f in frame_tensor_files]

        # Stack on the 0th dimension, from first to last.
        # dim: [FRAMES x CHANNELS x HEIGHT x WIDTH]
        frames_tensor = torch.vstack(frame_tensors)

        # Number of frames and labels (x, y, r) should be the same.
        assert frames_tensor.shape[0] == frame_labels.shape[0]

        # Scale the labels per the initialization arg; they were likely
        # constructed on larger images and need to be resized.
        scaled_frame_labels = frame_labels * self.labels_scale

        # Square pad the frames, and labels accordingly.
        padded_frames, padded_labels = square_pad_frame_and_labels(frames_tensor, scaled_frame_labels)

        # Resize the frames (and labels) to a square space.
        final_frames, final_labels = resize_frame_and_labels(padded_frames,
                                                             padded_labels,
                                                             self.resize_size)

        return final_frames, final_labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fove8n data loader test run')
    parser.add_argument('--root_videos_dir', required=True)
    parser.add_argument('--preprocessed_tensors_dir_pattern', required=True)
    parser.add_argument('--labels_scale', type=float, required=True)
    args = parser.parse_args()

    # all_set = args.root_videos_dir and args.preprocessed_tensors_dir_pattern and args.labels_scale
    # if not all_set:
    #     print("")
    print(args)

    train_dataset = MyDataset(root_videos_dir = args.root_videos_dir,
                              preprocessed_tensors_dir_pattern = args.preprocessed_tensors_dir_pattern,
                              labels_scale = args.labels_scale)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    print('Running one enumereation of the training dataloader.')
    for (idx, (input, labels)) in enumerate(train_dataloader):
        print('idx: {}\ninput.shape: {}\nlabels.shape: {}'.format(idx, input.shape, labels.shape))