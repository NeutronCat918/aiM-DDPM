import torch
import torch.nn as nn
import math
from einops.layers.torch import Rearrange
from einops import rearrange
from .utils import default


class DownSample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.net = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
            nn.Conv2d(dim * 4, default(dim_out, dim), 1),
        )

    def forward(self, x):
        return self.net(x)


class Upsample(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dim, dim_out or dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class ConvNextBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mult=2,
        time_embedding_dim=None,
        norm=True,
        group=8,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_embedding_dim, in_channels))
            if time_embedding_dim
            else None
        )

        self.in_conv = nn.Conv2d(
            in_channels, in_channels, 7, padding=3, groups=in_channels
        )

        self.block = nn.Sequential(
            nn.GroupNorm(1, in_channels) if norm else nn.Identity(),
            nn.Conv2d(in_channels, out_channels * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, out_channels * mult),
            nn.Conv2d(out_channels * mult, out_channels, 3, padding=1),
        )

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embedding=None):
        h = self.in_conv(x)
        if self.mlp is not None and time_embedding is not None:
            assert self.mlp is not None, "MLP is None"
            h = h + rearrange(self.mlp(time_embedding), "b c -> b c 1 1")
        h = self.block(h)
        return h + self.residual_conv(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FourierPosEmb(nn.Module):
    def __init__(self, dim, num_frequencies=16):
        """
        Args:
            dim: The output embedding dimension (must be an even number).
            num_frequencies: Number of Fourier frequencies to use (half of the output dimension).
        """
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even."
        self.dim = dim
        self.num_frequencies = num_frequencies
        self.frequencies = nn.Parameter(torch.randn(num_frequencies), requires_grad=True)  # Learnable frequencies

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size].
        Returns:
            Tensor of shape [batch_size, dim] with Fourier embeddings.
        """
        device = x.device
        x = x[:, None]  # Expand x to [batch_size, 1]
        
        # Apply Fourier transform with learnable frequencies
        freq_features = x * self.frequencies[None, :]
        
        # Compute sin and cos components
        sin_features = torch.sin(freq_features)
        cos_features = torch.cos(freq_features)

        # Concatenate sin and cos features
        emb = torch.cat((sin_features, cos_features), dim=-1)  # Shape: [batch_size, 2 * num_frequencies]

        # If dim > 2 * num_frequencies, pad with zeros to match the desired output dimension
        if emb.shape[-1] < self.dim:
            padding = self.dim - emb.shape[-1]
            emb = torch.cat((emb, torch.zeros((emb.shape[0], padding), device=device)), dim=-1)

        return emb



class BlockAttention(nn.Module):
    def __init__(self, gate_in_channel, residual_in_channel, scale_factor):
        super().__init__()
        self.gate_conv = nn.Conv2d(gate_in_channel, gate_in_channel, kernel_size=1, stride=1)
        self.residual_conv = nn.Conv2d(residual_in_channel, gate_in_channel, kernel_size=1, stride=1)
        self.in_conv = nn.Conv2d(gate_in_channel, 1, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        in_attention = self.relu(self.gate_conv(g) + self.residual_conv(x))
        in_attention = self.in_conv(in_attention)
        in_attention = self.sigmoid(in_attention)
        return in_attention * x