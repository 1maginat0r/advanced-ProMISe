import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import surface_distance
from surface_distance import metrics
from src.utils.util import save_predict, save_csv



def model_predict(args, img, prompt, img_encoder, prompt_encoder, mask_decoder):
    patch_size = args.rand_crop_size[0]
    device = args.device
    out = F.interpolate(img.float(), scale_factor=512 / patch_size, mode='trilinear')
    input_batch = out[0].transpose(0, 1)
    batch_features, feature_list = img_encoder(input_batch)
    feature_list.append(batch_features)
    #feature_list = feature_list[::-1]
    points_torch = prompt.transpose(0, 1)
    new_feature = []
    for i, (feature, feature_decoder) in enumerate(zip(feature_list, prompt_encoder)):
        if i == 3:
            new_feature.append(
                feature_decoder(feature.to(device), points_torch.clone(), [patch_size, patch_size, patch_size])
            )
        else:
            new_feature.a