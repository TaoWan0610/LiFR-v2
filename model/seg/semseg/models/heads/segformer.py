import torch
from torch import nn, Tensor
from typing import Tuple
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original
        self.activate = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        self.num_stages = len(dims)
        for i, dim in enumerate(dims):
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))

        self.linear_fuse = ConvModule(embed_dim*self.num_stages, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def fuse_features(self, features: Tuple[Tensor, ...]) -> Tensor:
        """Fuse the four-level pyramid at Stage-1 (1/4) resolution."""
        batch, _, height, width = features[0].shape
        outs = [
            self.linear_c1(features[0])
            .permute(0, 2, 1)
            .reshape(batch, -1, *features[0].shape[-2:])
        ]
        for index, feature in enumerate(features[1:], start=2):
            projected = (
                getattr(self, f"linear_c{index}")(feature)
                .permute(0, 2, 1)
                .reshape(batch, -1, *feature.shape[-2:])
            )
            outs.append(
                F.interpolate(
                    projected,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False,
                )
            )
        return self.linear_fuse(torch.cat(outs[::-1], dim=1))

    def classify_fused(self, fused_feature: Tensor, apply_dropout: bool = True) -> Tensor:
        """Classify an already fused 1/4-resolution decoder feature."""
        if apply_dropout:
            fused_feature = self.dropout(fused_feature)
        return self.linear_pred(fused_feature)

    def forward(self, features: Tuple[Tensor, ...]) -> Tensor:
        return self.classify_fused(self.fuse_features(features))
