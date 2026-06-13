import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from models_utils import Attention_SD,Attention_SD1
from typing import List, Tuple, Union


class SpatialAttention_max(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_max, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.GroupNorm(1,out_planes),
        nn.ReLU(),
    )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.GroupNorm(1,out_planes),
        nn.ReLU(),
    )


class MFFM(nn.Module):
    def __init__(self, dim_r,dim_d):
        super(MFFM, self).__init__()
        self.sa = SpatialAttention_max()
        self.conv1 = nn.Conv2d(dim_r+dim_d,dim_r,kernel_size=1,stride=1)
        self.ftt = Attention_SD(dim_r)
    def forward(self,r,d):

        x = torch.cat((r,d),1)
        sa = self.sa(x)
        out_r = r.mul(sa)
        out_d = d.mul(sa)
        out = self.ftt(out_r,out_d) + r

        return out.permute(0, 2, 3, 1).contiguous()


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels*2, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return out


class Adapter(nn.Module):
    def __init__(self, blk,task_num=None, bs = 1,DD=False) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.bs = bs
        self.prompt_learns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, 32),
                nn.GELU(),
                nn.Linear(32, dim),
                nn.GELU()
            ) for _ in range(8)
        ])
        self.task_num=task_num
        if DD:
            self.prompt_depth = nn.Sequential(
                nn.Linear(dim, 32),
                nn.GELU(),
                nn.Linear(32, dim),
                nn.GELU(),
            )
            self.prompt_mffm = MFFM(dim,dim)

        self.task_list = [ii * self.bs for ii in range(8)]
        self.dd = 0

    def forward(self, x,d=None):

        if d!=None:
            d = self.prompt_depth(d)
            x = self.prompt_mffm(x.permute(0, 3, 1, 2).contiguous(),d.permute(0, 3, 1, 2).contiguous())
        shorcut = x
        maps = []
        i = 0
        j = 0
        if self.task_num == None:
            for x1,x2 in zip(x,x):
                prompt = self.prompt_learns[i](x1.unsqueeze(0))
                maps.append(prompt)
                j = j + 1
                if j in self.task_list:
                    i = i + 1
            prompts = torch.cat(maps,0)
        else:
            for x1,x2 in zip(x,x):
                prompt = self.prompt_learns[self.task_num](x1.unsqueeze(0))
                maps.append(prompt)
            prompts = torch.cat(maps,0)


        promped = shorcut + prompts
        net = self.block(promped)
        return net

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.ft1 = Attention_SD1(64)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(self.ft1(x_cat, self.conv_res(x)))
        return x

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Conv2d(in_channels=144, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=144, kernel_size=3, padding=1),
            nn.BatchNorm2d(144),
            nn.GELU(),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

class SAM2UNet(nn.Module):
    def __init__(self, task_num=None,bs =1) -> None:
        super(SAM2UNet, self).__init__()
        model_cfg = "sam2_hiera_l.yaml"
        checkpoint_path = 'sam2_hiera_large.pt'
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)

        del model.memory_encoder
        # del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.image_encoder = model.image_encoder.trunk

        for param in self.image_encoder.parameters():
            param.requires_grad = False

        blocks = []
        numi = 0
        for block in self.image_encoder.blocks:
            if numi<8:
                blocks.append(
                    Adapter(block,task_num=task_num,bs =bs,DD=True)
                )
            else:
                blocks.append(
                    Adapter(block,task_num=task_num,bs =bs,DD=False)
                )
            numi=numi+1
        self.image_encoder.blocks = nn.Sequential(
            *blocks
        )
        self.rfb1 = RFB_modified(144,64)
        self.rfb2 = RFB_modified(288,64)
        self.rfb3 = RFB_modified(576,64)
        self.rfb4 = RFB_modified(1152,64)
        self.up1 = (Up(64, 64))
        self.up2 = (Up(64, 64))
        self.up3 = (Up(64, 64))

        self.head = nn.Conv2d(64, 1, kernel_size=1)
        self.dtor = nn.Sequential(
                nn.Conv2d(in_channels=144, out_channels=288, kernel_size=3,padding=1),
                nn.BatchNorm2d(288),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=288, out_channels=288, kernel_size=3, padding=1),
                nn.BatchNorm2d(288),
                nn.GELU(),
        )
        self.patch_embed = PatchEmbed(
            embed_dim=144,
        )
    def forward(self, x,d=None):

        d = self.patch_embed(d)
        d1 = self.dtor(d.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x1, x2, x3, x4 = self.image_encoder(x,d,d1)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        return out


