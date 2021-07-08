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
#         data = load_data_volume(
#             data=args.data,
#             data_dir=args.data_dir,
#             batch_size=args.batch_size,
#             augmentation=True,
#             split=split,
#             rand_crop_spatial_size=args.rand_crop_size,
#             num_worker=args.num_worker
#         )
#     elif split == 'val':
#         data = load_data_volume(
#             data=args.data,
#             data_dir=args.data_dir,
#             batch_size=1,
#             augmentation=False,
#             split=split,
#             deterministic=True,
#            