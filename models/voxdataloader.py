from tkinter import E
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import os
import sys
sys.path.append('.')
import math

####################################################################
## Please change the csvPath path to your path location 
####################################################################
class GazeFeatureDataloader(Dataset):

    def __init__(self, set_to_load="Training", stacked = False):
        
        assert set_to_load == "Training" or set_to_load == "Validation" or set_to_load == "Test"
        if set_to_load == "Training" or set_to_load == "Validation":
            csvPath = "voxdataset_train.csv"
            self.dpath = "<fill the training path here>"

        if set_to_load == "Test":
            csvPath = "voxdataset_test.csv"
            self.dpath = "<fill the test path here>"

        sample_list = pd.read_csv(csvPath)
        if set_to_load == "Training":
            num_rows = int(sample_list.shape[0] * 0.8)
            self.sample_list = sample_list.iloc[:num_rows].reset_index(drop=True)
        if set_to_load == "Validation":
            num_rows = int(sample_list.shape[0] * 0.8)
            self.sample_list = sample_list.iloc[num_rows:].reset_index(drop=True)
        if set_to_load == "Test":
            self.sample_list = sample_list

        self.set_to_load = set_to_load
        self.stacked = stacked

    def __getitem__(self, index):
        pdrow = self.sample_list.iloc[index, :]

        x = np.load(os.path.join(self.dpath, pdrow['id'], "tracklets", pdrow['video'], pdrow['tracklet']))['gaze_feature']
        if self.stacked:
            y = np.load(os.path.join(self.dpath, pdrow['id'], "tracklets", pdrow['video'], "processed", "iter_"+pdrow['tracklet']))['label']
        else:
            y = np.load(os.path.join(self.dpath, pdrow['id'], "tracklets", pdrow['video'], "processed", "pl_"+pdrow['tracklet']))['label']

        for i in range(y.shape[0]):
            if y[i] > 1: y[i] = 1

        x = x.T
        return torch.from_numpy(x).to(torch.float32), torch.from_numpy(np.array([y])).to(torch.long)

    def __len__(self):
        return self.sample_list.shape[0]




