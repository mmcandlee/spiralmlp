import os
import torch
import torch.nn as nn
from math import pi, sin, cos

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple

import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from semantic.custom_fun import load_checkpoint
    has_mmseg = True
except ImportError:
    print('Please Install mmsegmentation first for semantic segmentation.')
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    has_mmdet = True
except ImportError:
    print('Please Install mmdetection first for object detection and instance segmentation.')
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'spiral_S': _cfg(crop_pct=0.9),
    'spiral_M': _cfg(crop_pct=0.9),
    'spiral_L': _cfg(crop_pct=0.875),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SpiralFC(nn.Module):
    """
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        k = 2
        R = 14
        num_channels = [self.in_channels//k for _ in range(k)]
        
        for num_channel in num_channels:
            for i in range(num_channel):
                if i <= num_channel // 2:
                    offset[0, 2 * i + 0, 0, 0] = round(i*(R/num_channel) * cos(pi * i / 16))    # along h
                    offset[0, 2 * i + 1, 0, 0] = round(i*(R/num_channel) * sin(pi * i / 16))    # along w
                else:
                    offset[0, 2 * i + 0, 0, 0] = round((R - i*(R/num_channel)) * cos(pi * i / 16))    # along h
                    offset[0, 2 * i + 1, 0, 0] = round((R - i*(R/num_channel)) * sin(pi * i / 16))    # along w
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class PartitionPoolingLayer(torch.nn.Module):
    def __init__(self, num_paritions=1, size_paritions=None):
        super().__init__()
        self.size_paritions = size_paritions
        self.num_paritions = num_paritions
    def forward(self, x):
        ## input Batch, channels, height, width
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).view(b, h*w, c)
        size_paritions = self.size_paritions
        if size_paritions is None:
            size_paritions = c // self.num_paritions
        pool = torch.nn.functional.max_pool1d(x, kernel_size=size_paritions, stride=size_paritions)
        pool = pool.repeat_interleave(repeats=size_paritions, dim=-1)
        pool = pool.permute(0, 2, 1).view(b, c, h, w)
        return pool

class SpiralMLP(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.hrf = SpiralFC(dim, dim, (3, 3), 1, 0)

        self.reweight = Mlp(dim, dim // 4, dim * 2)

        self.pool = PartitionPoolingLayer(size_paritions=16)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x1 = self.pool(x.permute(0, 3, 1, 2))
        hrf = self.hrf(x1).permute(0, 2, 3, 1)
        c = self.mlp_c(x)

        a = (hrf + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = hrf * a[0] + c * a[1]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class SpiralBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=SpiralMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbedOverlapping(nn.Module):
    """ 2D Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=None, groups=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size
        # remove image_size in model init to support dynamic image size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    """ Downsample transition stage
    """
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x


def basic_blocks(dim, index, layers, mlp_ratio=3., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0., skip_lam=1.0, mlp_fn=SpiralMLP, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(SpiralBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))
    blocks = nn.Sequential(*blocks)

    return blocks


class SpiralNet(nn.Module):
    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0,
        qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., pretrained_cfg=None,
        norm_layer=nn.LayerNorm, mlp_fn=SpiralMLP, fork_feat=False):

        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=3, embed_dim=embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i+1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i+1], patch_size))

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self.cls_init_weights)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, SpiralFC):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out.permute(0, 3, 1, 2).contiguous())
        if self.fork_feat:
            return outs

        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        # B, H, W, C -> B, N, C
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x

        x = self.norm(x)
        cls_out = self.head(x.mean(1))
        return cls_out


@register_model
def SpiralMLP_B1(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = SpiralNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, mlp_fn=SpiralMLP, **kwargs)
    model.default_cfg = default_cfgs['spiral_S']
    return model


@register_model
def SpiralMLP_B2(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 3, 10, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = SpiralNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, mlp_fn=SpiralMLP, **kwargs)
    model.default_cfg = default_cfgs['spiral_S']
    return model


@register_model
def SpiralMLP_B3(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 4, 18, 3]
    mlp_ratios = [8, 8, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = SpiralNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, mlp_fn=SpiralMLP, **kwargs)
    model.default_cfg = default_cfgs['spiral_M']
    return model


@register_model
def SpiralMLP_B4(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 8, 27, 3]
    mlp_ratios = [8, 8, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = SpiralNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, mlp_fn=SpiralMLP, **kwargs)
    model.default_cfg = default_cfgs['spiral_L']
    return model


@register_model
def SpiralMLP_B5(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 4, 24, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [96, 192, 384, 768]
    model = SpiralNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, mlp_fn=SpiralMLP, **kwargs)
    model.default_cfg = default_cfgs['spiral_L']
    return model


if has_mmseg and has_mmdet:
    # For dense prediction tasks only
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class SpiralMLP_B1_feat(SpiralNet):
        def __init__(self, **kwargs):
            transitions = [True, True, True, True]
            layers = [2, 2, 4, 2]
            mlp_ratios = [4, 4, 4, 4]
            embed_dims = [64, 128, 320, 512]
            super(SpiralMLP_B1_feat, self).__init__(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                                                   mlp_ratios=mlp_ratios, mlp_fn=SpiralMLP, fork_feat=True)

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class SpiralMLP_B2_feat(SpiralNet):
        def __init__(self, **kwargs):
            transitions = [True, True, True, True]
            layers = [2, 3, 10, 3]
            mlp_ratios = [4, 4, 4, 4]
            embed_dims = [64, 128, 320, 512]
            super(SpiralMLP_B2_feat, self).__init__(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                                                    mlp_ratios=mlp_ratios, mlp_fn=SpiralMLP, fork_feat=True)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class SpiralMLP_B3_feat(SpiralNet):
        def __init__(self, **kwargs):
            transitions = [True, True, True, True]
            layers = [3, 4, 18, 3]
            mlp_ratios = [8, 8, 4, 4]
            embed_dims = [64, 128, 320, 512]
            super(SpiralMLP_B3_feat, self).__init__(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                                                    mlp_ratios=mlp_ratios, mlp_fn=SpiralMLP, fork_feat=True)

    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class SpiralMLP_B4_feat(SpiralNet):
        def __init__(self, **kwargs):
            transitions = [True, True, True, True]
            layers = [3, 8, 27, 3]
            mlp_ratios = [8, 8, 4, 4]
            embed_dims = [64, 128, 320, 512]
            super(SpiralMLP_B4_feat, self).__init__(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                                                    mlp_ratios=mlp_ratios, mlp_fn=SpiralMLP, fork_feat=True)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class SpiralMLP_B5_feat(SpiralNet):
        def __init__(self, **kwargs):
            transitions = [True, True, True, True]
            layers = [3, 4, 24, 3]
            mlp_ratios = [4, 4, 4, 4]
            embed_dims = [96, 192, 384, 768]
            super(SpiralMLP_B5_feat, self).__init__(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                                                    mlp_ratios=mlp_ratios, mlp_fn=SpiralMLP, fork_feat=True)
