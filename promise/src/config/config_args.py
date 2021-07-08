import argparse
import os
import warnings

parser = argparse.ArgumentParser()





# data
parser.add_argument("--data", default=None, type=str, choices=["kits", "pancreas", "lits", "colon"])
parser.add_argument("--save_dir", default="", type=str)
parser.add_argument("--data_dir", default="", type=str)
parser.add_argument("--num_worker", default=6, type=int)
parser.add_argument("--split", default="train", type=str)



# network
parser.add_argument("--lr", default=4e-4, type=float)
parser.add_argument("--device", default="cuda:0", type=str)
parser.add_argument("--max_epoch", default=200, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--rand_crop_size", default=(128, 128, 128), nargs='+', type=int)
parser.add_argument("--checkpoint", default="best", type=str)
parser.add_argument("--checkpoint_sam", default="./checkpoint_sam/sam_vit_b_01ec64.pth", type=str,
                    help='path of pretrained SAM')
parser.add_argument("--num_prompts", default=1, type=int)
parser.add_argument("--num_classes", default=2, type=int)
parser.add_argument("--tolerance", default=5, type=int)
parser.add_argument("--boundary_kernel_size", default=5, type=i