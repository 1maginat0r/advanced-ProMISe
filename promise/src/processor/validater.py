import torch
import torch.nn.functional as F
from src.utils.util import get_points
import numpy as np


def validater(args, val_data, logger, epoch_num,
          img_encoder, prompt_encoder_list, mask_decoder, loss_validation):
