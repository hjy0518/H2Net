import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.GroupNorm(1,out_planes),
        nn.ReLU(inplace=True),
    )

def conv3x3_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.ReLU(inplace=True),
    )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.GroupNorm(1,out_planes),
        nn.ReLU(inplace=True),
    )

def conv1x1_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.ReLU(inplace=True),
    )


def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor



class Attention_SD12(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(Attention_SD12, self).__init__()
        self.num_heads = num_heads

        self.qkv1conv_1 = conv1x1(dim,dim)
        self.qkv1conv_3 = conv1x1(dim,dim)
        self.qkv1conv_5 = conv1x1(dim,dim)

        self.qm = conv1x1(dim,dim)
        self.km = conv1x1(dim,dim)
        self.vm = conv1x1(dim,dim)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperatured = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_out = conv3x3_relu(dim * 2, dim)


    def forward(self, x,d):

        b, c, h, w = x.shape
        q_s = self.qkv1conv_5(x)
        k_s = self.qkv1conv_3(x)
        v_s = self.qkv1conv_1(x)
        q_s = torch.fft.fft2(q_s.float())
        k_s = torch.fft.fft2(k_s.float())
        v_s = torch.fft.fft2(v_s.float())
        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = custom_complex_normalization(attn_s, dim=-1)
        attn_s = torch.abs(torch.fft.ifft2(attn_s))


        dq_s = self.qm(d)
        dk_s = self.km(d)
        dv_s = self.vm(d)
        dq_s = rearrange(dq_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        dk_s = rearrange(dk_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        dv_s = rearrange(dv_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        dq_s = torch.nn.functional.normalize(dq_s, dim=-1)
        dk_s = torch.nn.functional.normalize(dk_s, dim=-1)
        dattn_s = (dq_s @ dk_s.transpose(-2, -1)) * self.temperatured
        dattn_s = torch.softmax(dattn_s, dim=-1)
        dattn_s = torch.fft.fft2(dattn_s.float())
        outr = torch.abs(torch.fft.ifft2(dattn_s @ v_s))
        outd = attn_s @ dv_s
        outd = rearrange(outd, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outr = rearrange(outr, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(torch.cat((outr,outd), 1))

        return out

class Attention_SD(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head = num_heads


        self.project_out = conv3x3_relu(dim * 2, dim)

        self.w1 = nn.Parameter(torch.tensor(1.0))
        self.w2 = nn.Parameter(torch.tensor(1.0))


        self.atten1 = Attention_SD12(dim)
        self.atten2 = Attention_SD12(dim)

    def forward(self, x, d):
        b, c, h, w = x.shape
        assert c % self.num_heads == 0, "dim 必须能被 num_heads 整除"

        # x <- d
        out_x2d = self.atten1(x,d)

        # d <- x
        out_d2x = self.atten2(d,x)

        w = torch.softmax(torch.stack([self.w1, self.w2]), dim=0)
        out = self.project_out(torch.cat([
            out_x2d,
            out_d2x,
        ], dim=1))

        return out



class Attention_SD1(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(Attention_SD1, self).__init__()
        self.num_heads = num_heads

        self.qkv1conv_1 = conv1x1(dim,dim)
        self.qkv1conv_3 = conv1x1(dim,dim)
        self.qkv1conv_5 = conv1x1(dim,dim)

        self.qm = conv1x1(dim,dim)
        self.km = conv1x1(dim,dim)
        self.vm = conv1x1(dim,dim)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperatured = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_out = conv3x3_bn_relu(dim * 4, dim)


    def forward(self, x,d):

        b, c, h, w = x.shape
        q_s = self.qkv1conv_5(x)
        k_s = self.qkv1conv_3(x)
        v_s = self.qkv1conv_1(x)
        q_s = torch.fft.fft2(q_s.float())
        k_s = torch.fft.fft2(k_s.float())
        v_s = torch.fft.fft2(v_s.float())
        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = custom_complex_normalization(attn_s, dim=-1)
        outr0 =  torch.abs(torch.fft.ifft2( attn_s @ v_s))
        attn_s = torch.abs(torch.fft.ifft2(attn_s))


        dq_s = self.qm(d)
        dk_s = self.km(d)
        dv_s = self.vm(d)
        dq_s = rearrange(dq_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        dk_s = rearrange(dk_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        dv_s = rearrange(dv_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        dq_s = torch.nn.functional.normalize(dq_s, dim=-1)
        dk_s = torch.nn.functional.normalize(dk_s, dim=-1)
        dattn_s = (dq_s @ dk_s.transpose(-2, -1)) * self.temperatured
        dattn_s = torch.softmax(dattn_s, dim=-1)
        outd0 = dattn_s @ dv_s
        dattn_s = torch.fft.fft2(dattn_s.float())

        outr = torch.abs(torch.fft.ifft2(dattn_s @ v_s))
        outd = attn_s @ dv_s

        outd = rearrange(outd, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outr = rearrange(outr, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outd0 = rearrange(outd0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outr0 = rearrange(outr0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(torch.cat((outr,outr0,outd,outd0), 1))

        return out





