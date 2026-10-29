import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom _pair utility to handle int/tuple conversion
def _pair(x):
    if isinstance(x, tuple):
        return x
    return (x, x)

# Define the Gaussian supervision function
class SupervisedGaussKernel2d(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, dilation=1, channel_wise=False):
        super(SupervisedGaussKernel2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.channel_wise = channel_wise
    
    def forward(self, feature1, feature2):
        """
        Perform pixel-adaptive convolution on feature1 with supervision from feature2.
        """
        bs, ch, h, w = feature1.shape
        
        # Unfold both feature1 and feature2
        f1_cols = F.unfold(feature1, self.kernel_size, self.dilation, self.padding, self.stride)
        f2_cols = F.unfold(feature2, self.kernel_size, self.dilation, self.padding, self.stride)

        f1_cols = f1_cols.view(bs, ch, self.kernel_size[0], self.kernel_size[1], h, w)
        f2_cols = f2_cols.view(bs, ch, self.kernel_size[0], self.kernel_size[1], h, w)

        center_y, center_x = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        
        # Use feature2 as the central guiding feature
        f2_center = f2_cols[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]

        # Compute squared differences between feature1 and feature2
        diff_sq = (f1_cols - f2_center).pow(2)
        if not self.channel_wise:
            diff_sq = diff_sq.sum(dim=1, keepdim=True)

        # Compute Gaussian kernel based on the difference
        gauss_kernel = torch.exp(-0.5 * diff_sq)

        # Apply Gaussian kernel to feature1
        supervised_feature1 = gauss_kernel * f1_cols

        # Reconstruct the output using fold (inverse of unfold)
        output = F.fold(supervised_feature1.view(bs, ch * self.kernel_size[0] * self.kernel_size[1], -1), 
                        (h, w), self.kernel_size, self.dilation, self.padding, self.stride)

        return output

# # Instantiate the module
# gauss_supervisor = SupervisedGaussKernel2d(kernel_size=3, stride=1, padding=1, dilation=1)

# # Example: using feature2 to supervise feature1
# feature1 = torch.randn(2, 64, 128, 128)  # Randomly initialized feature1 (batch size=2, 64 channels, 128x128 resolution)
# feature2 = torch.randn(2, 64, 128, 128)  # Randomly initialized feature2 as guiding feature

# # Apply supervision
# supervised_feature1 = gauss_supervisor(feature1, feature2)

# # Calculate loss between supervised_feature1 and feature2 (you can use other losses too)
# loss = F.mse_loss(supervised_feature1, feature2)

# # Backpropagate if part of a training loop
# loss.backward()
