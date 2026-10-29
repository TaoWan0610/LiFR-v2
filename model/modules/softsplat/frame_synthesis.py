import torch
import torch.nn as nn
from typing import Union
from .softsplat import softsplat
import numpy as np
from scipy.ndimage import gaussian_filter
from model.modules.memory_v1.blocks import MultiAttentionBlock
    
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (SE Block)"""
    def __init__(self, channels, reduction=4):
        """
        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for the bottleneck in the SE block.
        """
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.avg_pool(x)

        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)

        return x * scale

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        """
        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for the channel attention module.
            kernel_size: Convolution kernel size for the spatial attention module.
        """
        super(CBAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling

        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid_channel = nn.Sigmoid()

        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        channel_attention = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attention  # Apply channel attention

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))

        return x * spatial_attention  # Apply spatial attention

class Synthesis(torch.nn.Module):
    """Modified from the synthesis model in Softmax Splatting (https://github.com/sniklaus/softmax-splatting). Modifications:
    1) Warping only one frame with forward flow;
    2) Estimating the importance metric from the input frame and forward flow."""
    
    def __init__(
        self,
        feature_dims,
        activation='PReLU',
        warp_refine_identity_init=False,
        warp_refine_max_scale=1.0,
    ):
        super().__init__()

        if activation == 'PReLU':
            self.activation_layer = lambda out_channels: nn.PReLU(num_parameters=out_channels, init=0.25)
        elif activation == 'LeakyReLU':
            self.activation_layer = lambda out_channels: nn.LeakyReLU(negative_slope=0.01)
        elif activation == 'ELU':
            self.activation_layer = lambda out_channels: nn.ELU(alpha=1.0)
        elif activation == 'Mish':
            self.activation_layer = lambda out_channels: nn.Mish()
        elif activation == 'ReLU':
            self.activation_layer = lambda out_channels: nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        class Basic(torch.nn.Module):
            def __init__(self, strType, intChannels, boolSkip, skip_type='residual', attention_type='cbam', activation_layer=None):
                super().__init__()
                self.strType = strType
                self.activation_layer = activation_layer
                if strType == 'relu-conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        self.activation_layer(intChannels[0]),
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, padding=1, bias=False)
                    )
                elif strType == 'conv-relu-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'more-more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//4, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//4),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                    )
                elif strType == 'more-more-conv-k5':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=5, stride=1, padding=2, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=5, stride=1, padding=2, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//4, kernel_size=5, stride=1, padding=2, bias=False),
                        self.activation_layer(intChannels[1]//4),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//2, kernel_size=5, stride=1, padding=2, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=5, stride=1, padding=2, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=5, stride=1, padding=2, bias=False)
                    )
                elif strType == 'dilation-more-more-conv':
                    self.netMain = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1]//4, kernel_size=3, stride=1, padding=5, dilation=5, bias=False),
                        self.activation_layer(intChannels[1]//4),
                        torch.nn.Conv2d(in_channels=intChannels[1]//4, out_channels=intChannels[1]//2, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                        self.activation_layer(intChannels[1]//2),
                        torch.nn.Conv2d(in_channels=intChannels[1]//2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                        self.activation_layer(intChannels[1]),
                        torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=5, dilation=5, bias=False)
                    )

                elif strType == 'multi-attention':
                    dim = intChannels[2]
                    num_multi_attn = 1
                    num_heads = 16
                    LayerNorm_type = 'WithBias'
                    ffn_expansion_factor = 2.66
                    bias = False
                    self.is_DA=False
                    self.netMain = torch.nn.Sequential(
                        *[MultiAttentionBlock(dim, num_heads, LayerNorm_type, ffn_expansion_factor, bias, self.is_DA) 
                        for _ in range(num_multi_attn)]
                    )

                self.boolSkip = boolSkip

                if boolSkip == True:
                    if intChannels[0] == intChannels[2]:
                        self.netShortcut = None

                    elif intChannels[0] != intChannels[2]:
                        self.netShortcut = torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0, bias=False)

                self.skip_type = skip_type
                self.attention_type = attention_type
                if self.skip_type == 'dense':
                    self.matchChannels = torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=1, stride=1, padding=0, bias=False)
                elif self.skip_type == 'attention':
                    if self.attention_type == 'se':
                        self.se_block = SEBlock(intChannels[2])
                    elif self.attention_type == 'cbam':
                        self.cbam_block = CBAM(intChannels[2])



            def forward(self, tenInput, in1=None, in2=None):
                if self.strType == 'multi-attention':
                    if in1 is None:
                        in1 = tenInput
                    for layer in self.netMain:
                        tenMain = layer(tenMain, in1, None)
                    return tenMain
                tenMain = self.netMain(tenInput)

                if self.boolSkip == False:
                    return tenMain

                if self.skip_type == 'residual':
                    if self.netShortcut is None:
                        return tenMain + tenInput
                    else:
                        return tenMain + self.netShortcut(tenInput)

                elif self.skip_type == 'dense':
                    return self.dense_forward(tenInput)

                elif self.skip_type == 'attention':
                    if self.attention_type == 'se':
                        if self.netShortcut is None:
                            return self.se_block(tenMain) + tenInput
                        else:
                            return self.se_block(tenMain) + self.netShortcut(tenInput)
                    elif self.attention_type == 'cbam':
                        if self.netShortcut is None:
                            return self.cbam_block(tenMain) + tenInput
                        else:
                            return self.cbam_block(tenMain) + self.netShortcut(tenInput)

            def dense_forward(self, tenInput):
                """ Dense forward pass with DenseNet-style concatenation. """
                features = [tenInput]  # Initialize the feature list with the input
                for layer in self.netMain:
                    new_feature = layer(torch.cat(features, dim=1))  # Concatenate all previous features
                    features.append(new_feature)
                return self.matchChannels(torch.cat(features, dim=1))


        class Downsample(torch.nn.Module):
            def __init__(self, intChannels, activation_layer=None):
                super().__init__()
                self.activation_layer = activation_layer

                self.netMain = torch.nn.Sequential(
                    self.activation_layer(intChannels[0]),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=2, padding=1, bias=False),
                    self.activation_layer(intChannels[1]),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )

            def forward(self, tenInput):
                return self.netMain(tenInput)

        class Upsample(torch.nn.Module):
            def __init__(self, intChannels, activation_layer=None):
                super().__init__()
                self.activation_layer = activation_layer
                self.netMain = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    self.activation_layer(intChannels[0]),
                    torch.nn.Conv2d(in_channels=intChannels[0], out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                    self.activation_layer(intChannels[1]),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )

            def forward(self, tenInput):
                return self.netMain(tenInput)

        class Upsample_CrossScale(torch.nn.Module):
            def __init__(self, intChannels, upsample_mode='bilinear'):
                super().__init__()
                
                self.conv_high_res = torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[0], kernel_size=1, stride=1, padding=0, bias=False)
                self.upsample = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
                self.netMain = torch.nn.Sequential(
                    self.activation_layer(intChannels[0]*2),
                    torch.nn.Conv2d(in_channels=intChannels[0]*2, out_channels=intChannels[1], kernel_size=3, stride=1, padding=1, bias=False),
                    self.activation_layer(intChannels[1]),
                    torch.nn.Conv2d(in_channels=intChannels[1], out_channels=intChannels[2], kernel_size=3, stride=1, padding=1, bias=False)
                )

            def forward(self, high_res_feature, low_res_feature):
                high_res_feature = self.conv_high_res(high_res_feature)

                low_res_feature_upsampled = self.upsample(low_res_feature)

                tenInput = torch.cat([high_res_feature, low_res_feature_upsampled], 1)
                return self.netMain(tenInput)

        class Softmetric(torch.nn.Module):
            def __init__(self, skip_type, in_ch, out_ch, attention_type=None, activation_layer=None):
                super().__init__()
                embed_dim = [16, 32, 64, 96]

                self.netEventInput = torch.nn.Conv2d(in_channels=in_ch[0], out_channels=embed_dim[0]//2, kernel_size=3, stride=1, padding=1, bias=False)
                self.netFlow = torch.nn.Conv2d(in_channels=in_ch[1], out_channels=embed_dim[0]//2, kernel_size=3, stride=1, padding=1, bias=False)
                for intRow, intFeatures in enumerate(embed_dim):
                    self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [intFeatures, intFeatures, intFeatures], True, skip_type, attention_type, activation_layer))

                for intCol in [0]:
                    self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([embed_dim[0], embed_dim[1], embed_dim[1]], activation_layer=activation_layer))
                    self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([embed_dim[1], embed_dim[2], embed_dim[2]], activation_layer=activation_layer))
                    self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([embed_dim[2], embed_dim[3], embed_dim[3]], activation_layer=activation_layer))

                for intCol in [1]:
                    self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([embed_dim[3], embed_dim[2], embed_dim[2]], activation_layer=activation_layer))
                    self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([embed_dim[2], embed_dim[1], embed_dim[1]], activation_layer=activation_layer))
                    self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([embed_dim[1], embed_dim[0], embed_dim[0]], activation_layer=activation_layer))

                self.netOutput = Basic('conv-relu-conv', [embed_dim[0], embed_dim[0], out_ch], True, activation_layer=activation_layer)

            def forward(self, in1, in2):
                tenColumn = [None, None, None, None]

                tenColumn[0] = torch.cat([
                    self.netEventInput(in1),
                    self.netFlow(in2),
                ], 1)
                tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
                tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
                tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2])

                intColumn = 1
                for intRow in range(len(tenColumn) -1, -1, -1):
                    tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
                    if intRow != len(tenColumn) - 1:
                        tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])
                        if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[0, 0, 0, -1], mode='constant', value=0.0)

                        tenColumn[intRow] = tenColumn[intRow] + tenUp

                return self.netOutput(tenColumn[0])

        class Warp(torch.nn.Module):
            def __init__(
                self,
                embed_dim,
                activation_layer=None,
                identity_init=False,
                max_refine_scale=1.0,
            ):
                super().__init__()
                self.identity_init = bool(identity_init)
                self.max_refine_scale = float(max_refine_scale)
                self.refine_type = ['conv-relu-conv' if embed_dim[i] == 3 else 'more-more-conv' for i in range(len(embed_dim))]
                def make_refine_nets():
                    return nn.ModuleList([
                        Basic(
                            self.refine_type[i],
                            [embed_dim[i] + 1, embed_dim[i], embed_dim[i]],
                            True,
                            activation_layer=activation_layer,
                        )
                        for i in range(len(embed_dim))
                    ])

                self.nets = make_refine_nets()
                if self.identity_init:
                    self.refine_scales = nn.Parameter(torch.zeros(len(embed_dim)))

            def forward(self, tenEncone, tenMetricone, tenForward):
                tenOutput = []

                for intLevel in range(len(tenEncone)):
                    tenMetricone = torch.nn.functional.interpolate(input=tenMetricone, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False)
                    
                    tenForward = torch.nn.functional.interpolate(input=tenForward, size=(tenEncone[intLevel].shape[2], tenEncone[intLevel].shape[3]), mode='bilinear', align_corners=False) * (float(tenEncone[intLevel].shape[3]) / float(tenForward.shape[3]))
                    tenIn=torch.cat([tenEncone[intLevel], tenMetricone], 1)
                    tenWarp = softsplat(tenIn=tenIn, tenFlow=tenForward, tenMetric=tenMetricone, strMode='soft')
                    tenRefined = self.nets[intLevel](
                            tenWarp
                        )
                    if self.identity_init:
                        feature_channels = tenEncone[intLevel].shape[1]
                        tenWarpFeature = tenWarp[:, :feature_channels]
                        refine_scale = self.max_refine_scale * torch.tanh(
                            self.refine_scales[intLevel]
                        )
                        tenRefined = tenWarpFeature + refine_scale * (
                            tenRefined - tenWarpFeature
                        )
                    tenOutput.append(tenRefined)

                return tenOutput
        skip_type = 'attention'

        attention_type = 'cbam'

        self.netSoftmetric = Softmetric(skip_type, in_ch=[4,2], out_ch=1, attention_type=attention_type, activation_layer=self.activation_layer)

        self.netWarp = Warp(
            feature_dims,
            activation_layer=self.activation_layer,
            identity_init=warp_refine_identity_init,
            max_refine_scale=warp_refine_max_scale,
        )


    def forward(
        self,
        tenEncone,
        tenForward,
        event_voxel,
        second_anytime_flag=False,
        return_metric=False,
    ):
        times =4
        event_voxel = torch.nn.functional.interpolate(input=event_voxel, scale_factor=1/times, mode='bilinear', align_corners=False)
        tenForward = torch.nn.functional.interpolate(input=tenForward, scale_factor=1/times, mode='bilinear', align_corners=False) / times
        if second_anytime_flag==True:
            tenMetricone = torch.zeros(tenForward.shape[0], 1, tenForward.shape[2], tenForward.shape[3]).to(tenForward.device)
        else:
            tenMetricone = self.netSoftmetric(event_voxel, tenForward) * 2.0
        tenWarp = self.netWarp(tenEncone, tenMetricone, tenForward)
        if return_metric:
            return tenWarp, tenMetricone
        return tenWarp


@torch.no_grad()
def predict_tensor(src_frame: torch.Tensor, flow: torch.Tensor, model: Synthesis, batch_size: int = 32):
    out_frames = []
    for i in range(0, flow.shape[0], batch_size):
        bs = min(batch_size, flow.shape[0] - i)
        out_frames.append(model(src_frame.repeat(bs, 1, 1, 1), flow[i:i + bs]))
    out_frames = torch.cat(out_frames, dim=0)
    
    return out_frames

@torch.no_grad()
def softsplat_tensor(src_frame: torch.Tensor, flow: torch.Tensor, transforms, weight_type: Union[None, str] = None, return_tensor: bool = True):
    if weight_type == "flow_mag":
        weight = torch.sqrt(torch.square(flow[:, 0, :, :] + flow[:, 1, :, :])).unsqueeze(1)
        mode = "soft"
    else:
        weight = None
        mode = "avg"
    out_frames = softsplat(tenIn=src_frame.repeat(flow.shape[0], 1, 1, 1), tenFlow=flow, tenMetric=weight, strMode=mode)
    if return_tensor:
        return transforms.denormalize_frame(out_frames)
    else:
        return transforms.deprocess_frame(out_frames)
