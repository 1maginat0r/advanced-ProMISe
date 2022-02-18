import torch
import torch.nn as nn
import torch.nn.functional as F
class MLAHead(nn.Module):
    def __init__(self, mla_channels=256, mlahead_channels=128, norm_cfg=None):
        super(MLAHead, self).__init__()
        self.head2 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        self.head3 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        self.head4 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())
        self.head5 = nn.Sequential(nn.Conv3d(mla_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU(),
                     nn.Conv3d(mlahead_channels, mlahead_channels, 3, padding=1, bias=False),
                     nn.InstanceNorm3d(mlahead_channels),
                     nn.ReLU())

        self.up2 = nn.ConvTranspose3d(mlahead_channels, mlahead_channels, 2, stride=2)
        self.up3 = nn.ConvTranspose3d(mlahead_channels, mlahead_channels, 2, stride=2)
        self.up4 = nn.ConvTranspose3d(mlahead_channels, mlahead_channels, 2, stride=2)
        self.up5 = nn.ConvTranspose3d(mlahead_channels, mlahead_channels, 2, stride=2)


    def forward(self, mla_p2, mla_p3, mla_p4, mla_p5, scale_factor):
        # head2 = self.head2(mla_p2)
        head2 = self.up2(self.head2(mla_p2))
        head3 = self.up3(self.head3(mla_p3))
        head4 = self.up4(self.head4(mla_p4))
        head5 = self.up5(self.head5(mla_p5))
        return torch.cat([head2, head3, hea