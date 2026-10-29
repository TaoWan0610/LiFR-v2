import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock
from .utils import coords_grid, upflow_times, upflow8
from argparse import Namespace
import math
from contextlib import nullcontext
from typing import List
# from utils.image_utils import ImagePadder

try:
    # autocast = torch.cuda.amp.autocast
    autocast = torch.amp.autocast
    
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


def get_args():
    # This is an adapter function that converts the arguments given in out config file to the format, which the ERAFT
    # expects.
    args = Namespace(dropout=False,
                     mixed_precision=False,
                     clip=1.0)
    return args



class ERAFT(nn.Module):
    def __init__(self, n_first_channels, iters=2):
        # args:
        super(ERAFT, self).__init__()
        args = get_args()
        self.args = args
        self.iters = int(iters)
        self.force_fp16_inference = False
        # self.image_padder = ImagePadder(min_size=32)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        # feature network, context network, and update block
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn='instance', dropout=0,
            n_first_channels=n_first_channels,
        )
        self.cnet = BasicEncoder(
            output_dim=hdim + cdim, norm_fn='batch', dropout=0,
            n_first_channels=n_first_channels,
        )
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        # coords0 = coords_grid(N, H//8, W//8).to(img.device)
        # coords1 = coords_grid(N, H//8, W//8).to(img.device)
        coords0 = coords_grid(N, math.ceil(H / 8), math.ceil(W / 8), device=img.device, dtype=img.dtype)
        coords1 = coords_grid(N, math.ceil(H / 8), math.ceil(W / 8), device=img.device, dtype=img.dtype)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def extract_event2_context_pyramid(
        self,
        event2: torch.Tensor,
        detach: bool = True,
    ) -> List[torch.Tensor]:
        """Extract the frozen cnet semantic pyramid from one event window.

        ``event2`` uses the original event resolution.  The returned tensors
        are ``[layer1, layer2, layer3, relu(inp)]``.  For the default large
        ERAFT their channels are ``[64, 96, 128, 128]`` and their spatial
        strides relative to ``event2`` are approximately ``[4, 8, 16, 16]``.
        """
        if event2.ndim != 4:
            raise ValueError(
                f"event2 must be BCHW, got shape {tuple(event2.shape)}"
            )
        grad_context = torch.no_grad() if detach else nullcontext()
        with grad_context:
            # Keep this identical to the preprocessing in ERAFT.forward().
            event2_small = F.interpolate(
                event2,
                scale_factor=0.5,
                mode='bilinear',
                align_corners=False,
            )
            # Match ERAFT.forward(): with mixed_precision=False this nested
            # context deliberately keeps the frozen cnet in FP32 even when
            # the surrounding semantic adapter is trained under AMP.
            with autocast('cuda', enabled=self.args.mixed_precision):
                cnet_output, cnet_pyramid = self.cnet(
                    event2_small,
                    return_pyramid=True,
                )
            _, inp = torch.split(
                cnet_output,
                [self.hidden_dim, self.context_dim],
                dim=1,
            )
            features = [*cnet_pyramid, F.relu(inp)]

        if detach:
            features = [feature.detach() for feature in features]
        return features


    def forward(
        self,
        event1,
        event2,
        iters=None,
        flow_init=None,
        upsample=True,
        return_all=False,
        return_state=False,
        return_semantic_sources=False,
    ):
        """ Estimate optical flow between pair of frames """
        if iters is None:
            iters = self.iters
        # Pad Image (for flawless up&downsampling)
        # event1 = self.image_padder.pad(event1)
        # event2 = self.image_padder.pad(event2)

        # event1 = event1.contiguous()
        # event2 = event2.contiguous()

        # downsample event1 and event2
        times = 2
        event1 = F.interpolate(event1, scale_factor=1/times, mode='bilinear', align_corners=False)
        event2 = F.interpolate(event2, scale_factor=1/times, mode='bilinear', align_corners=False)

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast('cuda', enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([event1, event2])
        if not self.force_fp16_inference:
            fmap1 = fmap1.float()
            fmap2 = fmap2.float()

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast('cuda', enabled=self.args.mixed_precision):
            if return_semantic_sources:
                cnet, cnet_pyramid = self.cnet(event2, return_pyramid=True)
            else:
                cnet = self.cnet(event2)
                cnet_pyramid = None
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        # Initialize Grids. First channel: x, 2nd channel: y. Image is just used to get the shape
        coords0, coords1 = self.initialize_flow(event1)
        # import ipdb; ipdb.set_trace()

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        final_flow = None
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast('cuda', enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if return_all or itr == iters - 1:
                # Semantic training/inference only needs the final upsampled flow.
                # Flow-only training can set return_all=True for per-iteration supervision.
                if up_mask is None:
                    flow_up = upflow8(coords1 - coords0)
                else:
                    flow_up = self.upsample_flow(coords1 - coords0, up_mask)
                flow_up = upflow_times(flow_up, times)
            # The upsampled flow has twice the spatial size of the event input.
                _, _, H, W = event1.size()
                flow_up = F.interpolate(flow_up, size=(H*times, W*times), mode='bilinear', align_corners=False)

                if return_all:
                    flow_predictions.append(flow_up)
                else:
                    final_flow = flow_up
            # flow_predictions.append(self.image_padder.unpad(flow_up))

        if return_all:
            primary_output = flow_predictions
        elif return_state:
            primary_output = (coords1 - coords0, final_flow)
        else:
            primary_output = final_flow

        if not return_semantic_sources:
            return primary_output

        semantic_sources = {
            'cnet_pyramid': [*cnet_pyramid, inp],
            'fnet_event2': fmap2,
            'net_final': net,
        }
        return primary_output, semantic_sources
        # return coords1 - coords0, flow_predictions
