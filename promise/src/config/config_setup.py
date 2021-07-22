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
#             rand_crop_spatial_size=args.rand_crop_size,
#             num_worker=args.num_worker
#         )
#     else:
#         data = load_data_volume(
#         data=args.data,
#         batch_size=1,
#         data_dir=args.data_dir,
#         augmentation=False,
#         split=split,
#         rand_crop_spatial_size=args.rand_crop_size,
#         convert_to_sam=False,
#         do_test_crop=False,
#         deterministic=True,
#         num_worker=args.num_worker
#     )
#
#     return data


def load_data_set(args, split=''):

    if split == 'train':
        augmentation = True
        deterministic = False
    else:
        augmentation = False
        deterministic = True
        args.batch_size = 1

    data = load_data_volume(

        data=args.data,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augmentation=augmentation,
        split=split,
        deterministic=deterministic,
        rand_crop_spatial_size=args.rand_crop_size,
        num_worker=args.num_worker,
    )

    return data


def load_model(args, logger):
    if args.split == 'test':
        if args.use_pretrain:
            file_path = args.pretrain_path
            logger.info("- using pretrained model: {}".format(args.pretrain_path))
        else:
            if args.checkpoint == "last":
                file = "last.pth.tar"
            else:
                file = "best.pth.tar"
            file_path = os.path.join(args.save_dir, file)
            logger.info("- using pretrained model: {}".format(file_path))
        pretrained_model = torch.load(file_path, map_location='cpu')
    else:
        # please download pretrained SAM model (vit_b), and put it in the "/src/ckpl"
        sam = sam_model_registry["vit_b"](checkpoint=args.