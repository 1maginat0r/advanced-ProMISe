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
parser.add_argument("--max_