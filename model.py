import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from pvtv2 import pvt_v2_b2
from res2net import Res2Net50


def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential, nn.ModuleList)):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.GELU, nn.PReLU, nn.Softmax, nn.MaxPool2d, nn.AvgPool2d, nn.Dropout, nn.Identity, nn.UpsamplingBilinear2d)):
            pass
        else:
            m.initialize()

class Fusion(nn.Module):
    def __init__(self, channels):
        super(Fusion, self).__init__()
        self.linear2 = nn.Sequential(nn.Conv2d(channels[1], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear3 = nn.Sequential(nn.Conv2d(channels[2], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.linear4 = nn.Sequential(nn.Conv2d(channels[3], 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3, x4):
        x2, x3, x4   = self.linear2(x2), self.linear3(x3), self.linear4(x4)
        x4           = F.interpolate(x4, size=x2.size()[2:], mode='bilinear')
        x3           = F.interpolate(x3, size=x2.size()[2:], mode='bilinear')
        out          = torch.cat([x4, x3*x4, x2*x3*x4], dim=1)
        return out

    def initialize(self):
        weight_init(self)


class SegModel(nn.Module):
    def __init__(self, args):
        super(SegModel, self).__init__()
        self.backbone_name   = args.backbone
        if self.backbone_name =='res2net50':
            self.backbone    = Res2Net50()
            channels         = [256, 512, 1024, 2048]
        if self.backbone_name =='pvt_v2_b2':
            self.backbone    = pvt_v2_b2()
            channels         = [64, 128, 320, 512]

        self.fusion          = Fusion(channels)
        self.linear          = nn.Conv2d(64*3, 1, kernel_size=1)
        
        self.initialize()

    def forward(self, x):
        x1,x2,x3,x4 = self.backbone(x)
        pred        = self.fusion(x1,x2,x3,x4)
        pred        = self.linear(pred)
        return pred

    def initialize(self):
        weight_init(self)



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.initialize()
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    def initialize(self):
        weight_init(self)


class MultiAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(MultiAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.k = nn.Linear(dim, dim, bias=qkv_bias)
        # self.qv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.initialize()
    def forward(self, x, y):
        B, N, C = x.shape
        _, M, _ = y.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]   # make torchscript happy (cannot use tensor as tuple) 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    def initialize(self):
        weight_init(self)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.initialize()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def initialize(self):
        weight_init(self)




class LocalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, drop_path=0., mlp_ratio=2., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(LocalAttention, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, 1, 1, groups=1)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.initialize()

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x 

    def initialize(self):
        weight_init(self)


class GlobalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, drop_path=0., mlp_ratio=2., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(GlobalAttention, self).__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, 1, 1, groups=1)
        self.attn = MultiAttention(
            dim, num_heads=num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm12 = norm_layer(dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.initialize()

    def forward(self, x, y):
        x = x + self.pos_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        y = y.transpose(1,2)
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm12(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x 
    def initialize(self):
        weight_init(self)


class MixPool(nn.Module):
    def __init__(self, kernel):
        super(MixPool, self).__init__()
        self.kernel = kernel
        self.avgpool = nn.AvgPool2d(kernel_size=kernel)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel)
        self.projector = nn.Sequential(
                                nn.Conv2d(64*3*2, 64*3, kernel_size=1),
                                nn.BatchNorm2d(64*3),
                                nn.ReLU(inplace=True))
        self.initialize()
    def forward(self, x):
        out = torch.concat((self.avgpool(x), self.maxpool(x)), dim=1)
        out = self.projector(out)
        return out

    def initialize(self):
        weight_init(self)



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample = upsample
        self.initialize()
    def forward(self, x, skip=None):
        if self.upsample:
            x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    def initialize(self):
        weight_init(self)

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)
        self.initialize()
    
    def initialize(self):
        weight_init(self)

class ASTR(nn.Module):
    def __init__(self, args):
        super(ASTR, self).__init__()
        self.backbone_name   = args.backbone
        if self.backbone_name =='res2net50':
            self.backbone    = Res2Net50()
            channels         = [256, 512, 1024, 2048]
        if self.backbone_name =='pvt_v2_b2':
            self.backbone    = pvt_v2_b2()
            channels         = [64, 128, 320, 512]

        self.fusion          = Fusion(channels)
        pool                  = [MixPool(2**(i+1)) for i in range(args.clip_size-1)]
        self.pool            = nn.ModuleList(pool)
        conv                 =  [nn.Sequential(nn.Conv2d(64*3, 64*3, kernel_size=3, padding=1), nn.BatchNorm2d(64*3), nn.ReLU(inplace=True)) for i in range(args.clip_size)]
        self.conv            = nn.ModuleList(conv)
        body_attention      =   [nn.Sequential(
                                nn.Conv2d(64*3, 64*3, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(64*3),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(64*3, 1, kernel_size=1)) for i in range(args.clip_size-1)]
        self.body_attention  = nn.ModuleList(body_attention)
        self.projector       = nn.Conv2d(64*3*2, 64*3, kernel_size=1)
        # decoder              = [DecoderBlock(64*3+512, 256, False), DecoderBlock(256+256, 128), DecoderBlock(128, 64)]
        decoder              = [DecoderBlock(64*3*3+channels[1], channels[0], False), DecoderBlock(channels[0]+channels[0], channels[0]//2), DecoderBlock(channels[0]//2, channels[0]//4)]

        self.decoder         = nn.ModuleList(decoder)

        local_attention = [LocalAttention(64*3) for i in range(12)]
        self.local_attention = nn.ModuleList(local_attention)
        self.global_attention = GlobalAttention(64*3)
        # self.linear          = nn.Conv2d(channels[0]//4, 1, kernel_size=1)
        self.segmentation_head = SegmentationHead(channels[0]//4, 1, 1, 2)

        self.initialize()

    def forward(self, x):
        b, t, c, h, w   = x.shape
        x               = x.view(-1, c, h, w)
        x1,x2,x3,x4     = self.backbone(x)
        pred            = self.fusion(x1,x2,x3,x4)

        body            = pred.clone()
        # _,pc,ph,pw      = pred.shape
        # _,bc,bh,bw      = body.shape
        x1              = x1.view(b, t, -1, h//4, w//4)
        x2              = x2.view(b, t, -1, h//8, w//8)

        pred            = pred.view(b, t, -1, h//8, w//8)
        body            = body.view(b, t, -1, h//8, w//8)

        refs            = []
        bodymaps        = []
        for i in range(1, t):
            bodymap     = self.body_attention[i-1](body[:, i])
            ref         = self.pool[i-1](pred[:, i])
            ref         = self.conv[i](ref)
            ref         = torch.concat((pred[:, i], F.interpolate(ref, size=(h//8, w//8), mode='bilinear')), dim=1)
            ref         = self.projector(ref)
            ref         = [ref[j,:,bodymap[j, 0]>0].unsqueeze(0) for j in range(b)]
            refs.append(ref)
            bodymaps.append(bodymap.unsqueeze(1))

        local_pred      = self.local_attention[0](self.conv[0](pred[:, 0]))              #[C, H, W]
        for i in range(11):
            local_pred  = local_pred + self.local_attention[i+1](local_pred)
        
        global_pred_batch = []
        for i in range(b):
            ref_batch = [refs[j][i] for j in range(t-1)]
            ref_batch = torch.concat(ref_batch, dim=2)
            global_pred_batch.append(self.global_attention(local_pred[i].unsqueeze(0), ref_batch)) #[1, C, H ,W]
        global_pred = torch.concat(global_pred_batch, dim=0)
        global_pred = torch.cat((global_pred, local_pred, pred[:,0]), dim=1)
        # decoder
        out    = self.decoder[0](global_pred, x2[:,0])
        out    = self.decoder[1](out, x1[:,0])
        out    = self.decoder[2](out)
        out    = self.segmentation_head(out)
        return out, torch.concat(bodymaps, dim=1)

    def initialize(self):
        weight_init(self)

