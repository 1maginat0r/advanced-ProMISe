import torch
import torch.nn.functional as F
from src.utils.util import get_points
import numpy as np


def validater(args, val_data, logger, epoch_num,
          img_encoder, prompt_encoder_list, mask_decoder, loss_validation):
    patch_size = args.rand_crop_size[0]
    device = args.device
    with torch.no_grad():
        loss_summary = []
        for idx, (img, seg, spacing) in enumerate(val_data):
            out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
            input_batch = out.to(device)
            input_batch = input_batch[0].transpose(0, 1)
            batch_features, feature_list = img_encoder(input_batch)
            feature_list.append(batch_features)

            points_torch = get_points(args, seg, num_positive=10, num_negative=10