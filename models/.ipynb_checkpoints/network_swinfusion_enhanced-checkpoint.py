"""
SwinFusion Enhanced - 基于原始模型的优化版本
添加了:
1. CBAM (Convolutional Block Attention Module) 注意力机制
2. 边缘感知模块 (Edge-Aware Module)
3. SE (Squeeze-and-Excitation) 通道注意力
4. 多尺度特征融合模块

优化者: [你的名字]
日期: 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# ======================= 新增模块1: CBAM注意力 =======================
class ChannelAttention(nn.Module):
    """通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """CBAM: 结合通道和空间注意力"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ======================= 新增模块2: SE通道注意力 =======================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ======================= 新增模块3: 边缘感知模块 =======================
class EdgeAwareModule(nn.Module):
    """边缘感知模块 - 使用可学习的边缘提取"""
    def __init__(self, channels):
        super(EdgeAwareModule, self).__init__()

        # 可学习的边缘提取卷积
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False)
        )

        # 边缘特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # 残差缩放因子
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        edge = self.edge_conv(x)
        combined = torch.cat([x, edge], dim=1)
        enhanced = self.fusion(combined)
        return x + enhanced * self.scale


# ======================= 新增模块4: 多尺度特征融合 =======================
class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, channels):
        super(MultiScaleFusion, self).__init__()

        # 不同尺度的卷积
        self.conv1 = nn.Conv2d(channels, channels // 4, 1)
        self.conv3 = nn.Conv2d(channels, channels // 4, 3, padding=1)
        self.conv5 = nn.Conv2d(channels, channels // 4, 5, padding=2)
        self.conv7 = nn.Conv2d(channels, channels // 4, 7, padding=3)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        feat1 = self.conv1(x)
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        feat7 = self.conv7(x)

        multi_scale = torch.cat([feat1, feat3, feat5, feat7], dim=1)
        return self.fusion(multi_scale) + x


# ======================= 原始模型组件 (保持不变) =======================
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


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Cross_WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Cross_SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_A = norm_layer(dim)
        self.norm1_B = norm_layer(dim)
        self.attn_A = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_B = Cross_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path_A = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_B = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_A = norm_layer(dim)
        self.norm2_B = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_A = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_B = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, y, x_size):
        H, W = x_size
        B, L, C = x.shape

        shortcut_A = x
        shortcut_B = y
        x = self.norm1_A(x)
        y = self.norm1_B(y)
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        y_windows = window_partition(shifted_y, self.window_size)
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)

        if self.input_resolution == x_size:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.attn_mask)
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=self.attn_mask)
        else:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.calculate_mask(x_size).to(x.device))
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=self.calculate_mask(x_size).to(y.device))

        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, C)
        attn_windows_B = attn_windows_B.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_A, self.window_size, H, W)
        shifted_y = window_reverse(attn_windows_B, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            y = shifted_y
        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)

        x = shortcut_A + self.drop_path_A(x)
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))
        y = shortcut_B + self.drop_path_B(y)
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))
        return x, y


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Cross_BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            Cross_SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, y, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, y = checkpoint.checkpoint(blk, x, y, x_size)
            else:
                x, y = blk(x, y, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
            y = self.downsample(y)
        return x, y


class RSTB(nn.Module):
    """Residual Swin Transformer Block"""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim, input_resolution=input_resolution,
                                         depth=depth, num_heads=num_heads,
                                         window_size=window_size, mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path, norm_layer=norm_layer,
                                         downsample=downsample, use_checkpoint=use_checkpoint)

    def forward(self, x, x_size):
        return self.residual_group(x, x_size)


class CRSTB(nn.Module):
    """Cross Residual Swin Transformer Block"""
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(CRSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = Cross_BasicLayer(dim=dim, input_resolution=input_resolution,
                                         depth=depth, num_heads=num_heads,
                                         window_size=window_size, mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path, norm_layer=norm_layer,
                                         downsample=downsample, use_checkpoint=use_checkpoint)

        self.residual_group_A = BasicLayer(dim=dim, input_resolution=input_resolution,
                                         depth=depth, num_heads=num_heads,
                                         window_size=window_size, mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path, norm_layer=norm_layer,
                                         downsample=downsample, use_checkpoint=use_checkpoint)

        self.residual_group_B = BasicLayer(dim=dim, input_resolution=input_resolution,
                                         depth=depth, num_heads=num_heads,
                                         window_size=window_size, mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path, norm_layer=norm_layer,
                                         downsample=downsample, use_checkpoint=use_checkpoint)

    def forward(self, x, y, x_size):
        x = self.residual_group_A(x, x_size)
        y = self.residual_group_B(y, x_size)
        x, y = self.residual_group(x, y, x_size)
        return x, y


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])
        return x


# ======================= 增强版 SwinFusion 主模型 =======================
class SwinFusionEnhanced(nn.Module):
    """
    增强版 SwinFusion 模型

    相比原模型添加了:
    1. CBAM 注意力模块 - 在特征提取后增强重要特征
    2. 边缘感知模块 - 在浅层特征提取中保留边缘细节
    3. SE 通道注意力 - 增强通道特征选择
    4. 多尺度特征融合 - 融合不同尺度的特征
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=1,
                 embed_dim=96, Ex_depths=[4], Fusion_depths=[2, 2], Re_depths=[4],
                 Ex_num_heads=[6], Fusion_num_heads=[6, 6], Re_num_heads=[6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='',
                 resi_connection='1conv',
                 # ===== 新增参数: 控制优化模块 =====
                 use_cbam=True,           # 是否使用CBAM注意力
                 use_edge_aware=True,     # 是否使用边缘感知模块
                 use_se=True,             # 是否使用SE注意力
                 use_multi_scale=True,    # 是否使用多尺度融合
                 **kwargs):
        super(SwinFusionEnhanced, self).__init__()

        # 保存优化模块开关
        self.use_cbam = use_cbam
        self.use_edge_aware = use_edge_aware
        self.use_se = use_se
        self.use_multi_scale = use_multi_scale

        num_out_ch = in_chans
        self.img_range = img_range
        embed_dim_temp = int(embed_dim / 2)
        print('in_chans: ', in_chans)
        print('=== 增强模块配置 ===')
        print(f'  CBAM: {use_cbam}')
        print(f'  EdgeAware: {use_edge_aware}')
        print(f'  SE: {use_se}')
        print(f'  MultiScale: {use_multi_scale}')

        if in_chans == 3 or in_chans == 6:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        # ============ 浅层特征提取 (添加边缘感知) ============
        self.conv_first1_A = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first1_B = nn.Conv2d(in_chans, embed_dim_temp, 3, 1, 1)
        self.conv_first2_A = nn.Conv2d(embed_dim_temp, embed_dim, 3, 1, 1)
        self.conv_first2_B = nn.Conv2d(embed_dim_temp, embed_dim_temp, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 【新增】边缘感知模块
        if self.use_edge_aware:
            self.edge_aware_A = EdgeAwareModule(embed_dim)
            self.edge_aware_B = EdgeAwareModule(embed_dim)

        # ============ 深度特征提取 ============
        self.Ex_num_layers = len(Ex_depths)
        self.Fusion_num_layers = len(Fusion_depths)
        self.Re_num_layers = len(Re_depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.softmax = nn.Softmax(dim=0)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr_Ex = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Ex_depths))]
        dpr_Fusion = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Fusion_depths))]
        dpr_Re = [x.item() for x in torch.linspace(0, drop_path_rate, sum(Re_depths))]

        # 特征提取层A
        self.layers_Ex_A = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=Ex_depths[i_layer],
                         num_heads=Ex_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection)
            self.layers_Ex_A.append(layer)
        self.norm_Ex_A = norm_layer(self.num_features)

        # 特征提取层B
        self.layers_Ex_B = nn.ModuleList()
        for i_layer in range(self.Ex_num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=Ex_depths[i_layer],
                         num_heads=Ex_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Ex[sum(Ex_depths[:i_layer]):sum(Ex_depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection)
            self.layers_Ex_B.append(layer)
        self.norm_Ex_B = norm_layer(self.num_features)

        # 【新增】CBAM注意力 (特征提取后)
        if self.use_cbam:
            self.cbam_A = CBAM(embed_dim)
            self.cbam_B = CBAM(embed_dim)

        # 【新增】SE注意力
        if self.use_se:
            self.se_A = SEBlock(embed_dim)
            self.se_B = SEBlock(embed_dim)

        # 融合层
        self.layers_Fusion = nn.ModuleList()
        for i_layer in range(self.Fusion_num_layers):
            layer = CRSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=Fusion_depths[i_layer],
                         num_heads=Fusion_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Fusion[sum(Fusion_depths[:i_layer]):sum(Fusion_depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection)
            self.layers_Fusion.append(layer)
        self.norm_Fusion_A = norm_layer(self.num_features)
        self.norm_Fusion_B = norm_layer(self.num_features)

        # 【新增】多尺度特征融合
        if self.use_multi_scale:
            self.multi_scale_fusion = MultiScaleFusion(embed_dim * 2)

        # 重建层
        self.layers_Re = nn.ModuleList()
        for i_layer in range(self.Re_num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0], patches_resolution[1]),
                         depth=Re_depths[i_layer],
                         num_heads=Re_num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr_Re[sum(Re_depths[:i_layer]):sum(Re_depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection)
            self.layers_Re.append(layer)
        self.norm_Re = norm_layer(self.num_features)

        # 卷积层
        self.conv_after_body_Fusion = nn.Conv2d(2 * embed_dim, embed_dim, 3, 1, 1)
        self.conv_last1 = nn.Conv2d(embed_dim, embed_dim_temp, 3, 1, 1)
        self.conv_last2 = nn.Conv2d(embed_dim_temp, int(embed_dim_temp/2), 3, 1, 1)
        self.conv_last3 = nn.Conv2d(int(embed_dim_temp/2), num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features_Ex_A(self, x):
        x = self.lrelu(self.conv_first1_A(x))
        x = self.lrelu(self.conv_first2_A(x))

        # 【新增】边缘感知
        if self.use_edge_aware:
            x = self.edge_aware_A(x)

        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Ex_A:
            x = layer(x, x_size)

        x = self.norm_Ex_A(x)
        x = self.patch_unembed(x, x_size)

        # 【新增】CBAM注意力
        if self.use_cbam:
            x = self.cbam_A(x)

        # 【新增】SE注意力
        if self.use_se:
            x = self.se_A(x)

        return x

    def forward_features_Ex_B(self, x):
        x = self.lrelu(self.conv_first1_A(x))
        x = self.lrelu(self.conv_first2_A(x))

        # 【新增】边缘感知
        if self.use_edge_aware:
            x = self.edge_aware_B(x)

        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Ex_B:
            x = layer(x, x_size)

        x = self.norm_Ex_B(x)
        x = self.patch_unembed(x, x_size)

        # 【新增】CBAM注意力
        if self.use_cbam:
            x = self.cbam_B(x)

        # 【新增】SE注意力
        if self.use_se:
            x = self.se_B(x)

        return x

    def forward_features_Fusion(self, x, y):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        y = self.patch_embed(y)
        if self.ape:
            x = x + self.absolute_pos_embed
            y = y + self.absolute_pos_embed
        x = self.pos_drop(x)
        y = self.pos_drop(y)

        for layer in self.layers_Fusion:
            x, y = layer(x, y, x_size)

        x = self.norm_Fusion_A(x)
        x = self.patch_unembed(x, x_size)
        y = self.norm_Fusion_B(y)
        y = self.patch_unembed(y, x_size)

        x = torch.cat([x, y], 1)

        # 【新增】多尺度特征融合
        if self.use_multi_scale:
            x = self.multi_scale_fusion(x)

        x = self.lrelu(self.conv_after_body_Fusion(x))
        return x

    def forward_features_Re(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_Re:
            x = layer(x, x_size)

        x = self.norm_Re(x)
        x = self.patch_unembed(x, x_size)

        x = self.lrelu(self.conv_last1(x))
        x = self.lrelu(self.conv_last2(x))
        x = self.conv_last3(x)
        return x

    def forward(self, A, B):
        x = A
        y = B
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        y = self.check_image_size(y)

        self.mean_A = self.mean.type_as(x)
        self.mean_B = self.mean.type_as(y)
        self.mean = (self.mean_A + self.mean_B) / 2

        x = (x - self.mean_A) * self.img_range
        y = (y - self.mean_B) * self.img_range

        # 前向传播
        x = self.forward_features_Ex_A(x)
        y = self.forward_features_Ex_B(y)
        x = self.forward_features_Fusion(x, y)
        x = self.forward_features_Re(x)

        x = x / self.img_range + self.mean
        return x[:, :, :H*self.upscale, :W*self.upscale]


# ======================= 测试代码 =======================
if __name__ == '__main__':
    print("测试增强版 SwinFusion 模型...")

    model = SwinFusionEnhanced(
        img_size=64,
        in_chans=1,
        embed_dim=60,
        Ex_depths=[4],
        Fusion_depths=[2, 2],
        Re_depths=[4],
        Ex_num_heads=[6],
        Fusion_num_heads=[6, 6],
        Re_num_heads=[6],
        window_size=8,
        use_cbam=True,
        use_edge_aware=True,
        use_se=True,
        use_multi_scale=True
    )

    # 测试输入
    x1 = torch.randn(1, 1, 64, 64)
    x2 = torch.randn(1, 1, 64, 64)

    # 前向传播
    with torch.no_grad():
        out = model(x1, x2)

    print(f"输入尺寸: {x1.shape}")
    print(f"输出尺寸: {out.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("测试通过!")