from __future__ import print_function, division
import os
from abc import ABC

import torch
import pandas as pd
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()


class SandedAcrylicsDataset(Dataset):
    def __init__(self, json_file, images_dir, transform=None):
        """
        json_file (json file): Path to json file containing image_names with corresponding labels
        root_dir (directory): The folder containing all the images
        transform (callable, optional): The transform to be applied on a image sample if given
        """
        self.sandingFrame = pd.read_json(json_file)
        self.imageDir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.sandingFrame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgName = os.path.join(self.imageDir, self.sandingFrame.iloc[idx, 0])
        image = io.imread(imgName)
        label = np.array(self.sandingFrame.iloc[idx, 1]).astype('float')
        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        return sample



