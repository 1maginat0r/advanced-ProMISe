import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
import math
from segment_anything.modeling.common import MLPBlock
from segment_anything.modeling.image_encoder import PatchEmbed, window_partition, window_unpartition

class Adapter(nn.Module):
    def __init__(
            self,
            input_dim,
            mid_dim
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, mid_dim)
        self.conv = nn.Conv3d(in_channels = mid_dim, out_channels = mid_dim, kernel_size=3, padding=1, groups=mid_dim)
        self.linear2 = nn.Linear(mid_dim, input_dim)

    def forward(self, features):
        out = self.linear1(features)
        out = F.relu(out)
        out = out.permute(0, 4, 1, 2, 3)
        out = self.conv(out)
        out = out.permute(0, 2, 3, 4, 1)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = features + out
        return out

class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class Promise(nn.Module):
    def __init__(
            self,
            img_size: int = 1024,
            patch_size: int = 16,
            patch_depth: int = 32,
            #patch_depth: int = 32,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            out_chans: int = 256,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_abs_pos: bool = True,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            cubic_window_size: int = 0,
            global_attn_indexes: Tuple[int, ...] = (),
            num_slice = 1
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size
        self.patch_depth = patch_depth
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_slice = num_slice
        if self.num_slice > 1:
            self.slice_embed = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim,
                                         kernel_size=(1,1,self.num_slice), stride=(1,1,self.num_slice),
                                         groups=embed_dim)

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )
            self.depth_embed = nn.Parameter(
                torch.ones(1, patch_depth, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block_3d(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=cubic_window_size,
                res_size=window_size if i not in global_attn_indexes else img_size // patch_size,
                shift=cubic_window_size // 2 if i % 2 == 0 else 0,
                depth=self.patch_depth
            )
            self.blocks.append(block)

        self.neck_3d = nn.ModuleList()
        for i in range(4):
            self.neck_3d.append(nn.Sequential(
                nn.Conv3d(768, out_chans, 1, bias=False),
                LayerNorm3d(out_chans),
                nn.Conv3d(
                    out_chans,
                    out_chans,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                LayerNorm3d(out_chans),
            ))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # patch embedding
        with torch.no_grad():
            x = self.patch_embed(x)

        if self.num_slice > 1:
            x = self.slice_embed(x.permute(3, 1, 2, 0).unsqueeze(0))
            x = x.permute(0, 2, 3, 4, 1)
        else:
            x = x.permute(1, 2, 0, 3).unsqueeze(0)

        x = x.permute(0, 4, 1, 2, 3)
        # x = self.m1(x)
        # x = self.m2(x)
        x = x.permute(0, 2, 3, 4, 1)

        # position embedding
        if self.pos_embed is not None:
            pos_embed = F.avg_pool2d(self.pos_embed.permute(0,3,1,2), kernel_size=int(64 / self.patch_depth)).permute(0,2,3,1).unsqueeze(3)
            pos_embed = pos_embed + (self.depth_embed.unsqueeze(1).unsqueeze(1))
            x = x + pos_embed

        idx = 0
        feature_list = []
        for blk in self.blocks[:6]:
            x = blk(x)
            idx += 1
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx//3-1](x.permute(0, 4, 1, 2, 3)))
        for blk in self.blocks[6:12]:
            x = blk(x)
            idx += 1
            if idx % 3 == 0 and idx != 12:
                feature_list.append(self.neck_3d[idx//3-1](x.permute(0, 4, 1, 2, 3)))

        x = self.neck_3d[-1](x.permute(0, 4, 1, 2, 3))

        return x, feature_list





class Block_3d(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            act_layer: Type[nn.Module] = nn.GELU,
            use_rel_pos: bool = False,
            rel_pos_zero_init: bool = True,
            window_size: int = 0,
            res_size = None,
            shift = None,
            depth = 32,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
        