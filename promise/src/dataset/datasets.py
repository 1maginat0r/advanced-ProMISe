
import pickle
import os, sys
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from .base_dataset import BaseVolumeDataset


class KiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):