import os

import numpy as np
import torch
from torch.utils.data import Dataset

class BlossomDataset(Dataset):

    def __init__(self, file_list, blossom_dir = 'data/blossom_cache', weather_dir = 'data/weather_cache'):
        self.file_list = file_list

        self.blossom_dir = blossom_dir
        self.weather_dir = weather_dir

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        blossom = torch.Tensor(np.load(os.path.join(self.blossom_dir, self.file_list[index])))
        weather = torch.Tensor(np.load(os.path.join(self.weather_dir, self.file_list[index])))
        return blossom, weather, self.file_list[index]