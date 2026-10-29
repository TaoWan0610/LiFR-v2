"""RGB backbone used by LiFR and its distillation teacher."""
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from model.modules.layers import DropPath

class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W, metric: Tensor=None) -> Tensor:
        B, N, C = x.shape
        if metric is None:
            q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)
        else:
            q = self.q(metric).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))

class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)    # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x: Tensor, H, W, metric: Tensor=None) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, metric))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

ENCODER_VARIANTS = {
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]],
}


class Encoder(nn.Module):
    """Four-stage RGB features, with checkpoint-compatible stage names."""

    def __init__(self, model_name='B2', modals=None, with_events=False):
        super().__init__()
        if model_name not in ENCODER_VARIANTS:
            raise ValueError(f'Unknown Encoder variant: {model_name!r}')
        if with_events:
            raise ValueError('Encoder is the RGB backbone; event features are encoded by EGCM.')
        modals = ['img'] if modals is None else list(modals)
        if len(modals) > 1:
            raise ValueError('The RGB backbone takes one image modality.')
        self.modals = []
        self.num_modals = 0
        self.with_events = False
        embed_dims, depths = ENCODER_VARIANTS[model_name]
        self.channels = embed_dims

        self.patch_embed1 = PatchEmbed(3, embed_dims[0], 7, 4, 7 // 2)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], 3, 2, 3 // 2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], 3, 2, 3 // 2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], 3, 2, 3 // 2)
        drop_paths = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        cursor = 0
        for stage, (channels, depth, heads, sr_ratio) in enumerate(
            zip(embed_dims, depths, [1, 2, 5, 8], [8, 4, 2, 1]), start=1
        ):
            setattr(self, f'block{stage}', nn.ModuleList([
                Block(channels, heads, sr_ratio, drop_paths[cursor + index])
                for index in range(depth)
            ]))
            setattr(self, f'norm{stage}', nn.LayerNorm(channels))
            cursor += depth

    def forward(self, inputs, metric=None):
        image = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        batch = image.shape[0]
        features = []
        for stage in range(1, 5):
            tokens, height, width = getattr(self, f'patch_embed{stage}')(image)
            query = None if metric is None else metric[stage - 1].flatten(2).transpose(1, 2)
            for block in getattr(self, f'block{stage}'):
                tokens = block(tokens, height, width, query)
            image = getattr(self, f'norm{stage}')(tokens)
            image = image.reshape(batch, height, width, -1).permute(0, 3, 1, 2)
            features.append(image)
        return features
