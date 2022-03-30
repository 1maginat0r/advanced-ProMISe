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
            new_feature.append(feature.to(device))
    img_resize = F.interpolate(img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device), scale_factor=64/patch_size,
                               mode="trilinear")
    new_feature.append(img_resize)
    masks = mask_decoder(new_feature, 2, patch_size//64)
    masks = masks.permute(0, 1, 4, 2, 3)
    return masks

def get_points_prompt(args, points_dict, cumulative=False):
    """
    get prompt tensor (input) with given point-locations
    """
    patch_size = args.rand_crop_size[0]

    # the first point is always same, and we use it as anchor point, i.e. (x[0], y[0], z[0]) --> (206, 126, 320)
    # manually test with 1 prompt, 5 prompts and 10 prompts
    x, y, z = points_dict['x_location'], points_dict['y_location'], points_dict['z_location'] # x with size--> tensor([num_prompts, 1])

    x_m = (torch.max(x) + torch.min(x)) // 2
    y_m = (torch.max(y) + torch.min(y)) // 2
    z_m = (torch.max(z) + torch.min(z)) // 2

    # considered transpose in dataloader, e.g. match to real original images
    d_min = x_m - patch_size // 2
    d_max = x_m + patch_size // 2
    h_min = z_m - patch_size // 2
    h_max = z_m + patch_size // 2
    w_min = y_m - patch_size // 2
    w_max = y_m + patch_size // 2



    points = torch.cat([z - d_min, x - w_min, y - h_min], dim=1).unsqueeze(1).float()
    points_torch = points.to(args.device)

    patch_dict = {'w_min': w_min, 'w_max': w_max, 'h_min': h_min, 'h_max': h_max, 'd_min': d_min, 'd_max': d_max}

    return points_torch, patch_dict



def get_final_prediction(args, img, seg_dict, points_dict, img_encoder, prompt_encoder_list, mask_decoder):
    seg = seg_dict['seg']

    device = args.device
    patch_size = args.rand_crop_size[0]

    points_torch, patch_dict = get_points_prompt(args, points_dict)


    w_min, w