import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from torch.nn.modules.normalization import LayerNorm
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


class PatchEmbedding(nn.Module):
    def __init__(
        self, img_size: int = 224, patch_size: int = 16, in_chans=3, embed_dim=768
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.projection = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, patch_size, patch_size),
            Rearrange("b e h w -> b (h w) e"),
        )
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.positions = nn.Parameter(
            torch.rand(1, ((img_size[0] // patch_size[0]) ** 2) + 1, embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = self.projection(x)

        cls_tokens = repeat(self.cls_token, "1 n e -> b n e", b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positions

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.lin = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        x = self.lin(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim=768, num_heads=8, qkv_bias=False, attn_drop=0.0, out_drop=0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Sequential(
            nn.Linear(dim, dim * 3, bias=qkv_bias),
            Rearrange(
                "b n (num_qkv heads e) -> b num_qkv heads n e",
                num_qkv=3,
                heads=num_heads,
            ),
        )
        self.attn = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.out = nn.Sequential(
            Rearrange("b h n e -> b n (h e)"),
            nn.Linear(in_features=dim, out_features=dim),
        )
        self.out_drop = nn.Dropout(p=out_drop)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)
        q, k, v = q.squeeze(1), k.squeeze(1), v.squeeze(1)
        attent = self.attn((q @ torch.transpose(k, dim0=2, dim1=3)) * self.scale)
        attent = self.attn_drop(attent)

        out = self.out(attent @ v)
        x = self.out_drop(out)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, drop_rate=0.0, qkv_bias=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=drop_rate,
            out_drop=drop_rate,
            qkv_bias=qkv_bias,
        )

        self.drop = nn.Dropout(p=drop_rate)

        self.mlp = MLP(in_features=dim, hidden_features=dim * mlp_ratio, drop=drop_rate)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))

        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
        self, depth, dim, num_heads=8, mlp_ratio=4, drop_rate=0.0, qkv_bias=False
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(dim, num_heads, mlp_ratio, drop_rate, qkv_bias=qkv_bias)
                for i in range(depth)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        drop_rate=0.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.transformer = Transformer(
            depth=depth,
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            qkv_bias=qkv_bias,
        )

        self.classifier = nn.Sequential(
            LayerNorm(embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes),
        )

    def forward(self, x):
        embed = self.patch_embed(x)

        transformer = self.transformer(embed)

        x = self.classifier(transformer[:, 0, :])

        return x
