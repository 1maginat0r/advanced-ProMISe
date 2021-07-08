from src.dataset.datasets import load_data_volume
from src.models.image_encoder import Promise
from src.models.prompt_encoder import PromptEncoder, TwoWayTransformer
from src.models.mask_decoder import VIT_MLAHead
import os
import torch
from functools import partial
import torch.nn as nn
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torch.optim import AdamW
from monai.losses import DiceCELoss, DiceLoss
# def load_data_set(args, split=''):
#     if split == 'train':
# 