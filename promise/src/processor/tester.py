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


    w_min, w_max = patch_dict['w_min'], patch_dict['w_max']
    h_min, h_max = patch_dict['h_min'], patch_dict['h_max']
    d_min, d_max = patch_dict['d_min'], patch_dict['d_max']



    w_l = max(0, -w_min)
    w_r = max(0, w_max - points_dict['z_dimension'])
    h_l = max(0, -h_min)
    h_r = max(0, h_max - points_dict['y_dimension'])
    d_l = max(0, -d_min)
    d_r = max(0, d_max - points_dict['x_dimension'])

    d_min = max(0, d_min)
    h_min = max(0, h_min)
    w_min = max(0, w_min)

    img_patch = img[:, :, d_min:d_max, h_min:h_max, w_min:w_max].clone()
    # the pad follows the format (pad ordering) of torch
    # pad deals the points that are not included in the original patch (default size: 128^3)
    # if such case happens, the points values are negative
    # global query in prompt encoder is a learnable variable
    # this may because the point values represent the coordinate information with a shifted origin (-d_min, -h_min, -w_min)
    img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r)) # follow the format of torch pad function

    pred = model_predict(args,
                         img_patch,
                         points_torch,
                         img_encoder,
                         prompt_encoder_list,
                         mask_decoder)
    pred = pred[:, :, d_l:patch_size - d_r, h_l:patch_size - h_r, w_l:patch_size - w_r]
    pred = F.softmax(pred, dim=1)[:, 1]
    # seg_pred is not meta tensor
    # seg_pred = torch.zeros(1, 1, points_dict['x_dimension'], points_dict['z_dimension'], points_dict['y_dimension']).to(device)
    # zeros_like carries meta tensor
    seg_pred = torch.zeros_like(img).to(device)[:, 0, :].unsqueeze(0)
    seg_pred[:, :, d_min:d_max, h_min:h_max, w_min:w_max] += pred

    final_pred = F.interpolate(seg_pred, size=seg.shape[2:], mode="trilinear")
    img_orig = F.interpolate(img, size=seg.shape[2:], mode="trilinear")

    # original_image = F.pad(input=img, pad=(0, 1, 0, 0), mode='constant', value=0)
    return final_pred, img_orig


def get_points(prompt, sample):
    z = torch.where(prompt == 1)[3][sample].unsqueeze(1)  # --> tensor([[x_value]]) instead of tensor([x_value])
    x = torch.where(prompt == 1)[2][sample].unsqueeze(1)  # ignore: check datasets.py, --> self.spatial_index
    y = torch.where(prompt == 1)[4][sample].unsqueeze(1)
    # consider x,y,z here is a,b,c. and abc need to match features after img_encoder with size (b,c,x,z,y)
    # e.g. a-->x, b-->z, c-->y
    # xyz are the coordinates after dataloader, e.g. without consider the self.spatial_index
    return x, y, z
def get_points_location(args, prompt):
    """
    use this to get anchor points
    """
    l = len(torch.where(prompt == 1)[0])
    sample = np.random.choice(np.arange(l), args.num_prompts, replace=True)
    x, y, z = get_points(prompt, sample)


    # this x, y, z location follows the original after change spatial_index
    points_dict = {'x_location': x, 'y_location': y, 'z_location': z,
                   'x_dimension': prompt.shape[2], 'y_dimension': prompt.shape[3], 'z_dimension':