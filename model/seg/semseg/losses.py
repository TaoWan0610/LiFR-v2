import torch
from torch import nn, Tensor
from torch.nn import functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def outlier_penalty_loss(X, r=3):
    """
    Compute the outlier penalty loss (OPL) for a given latent variable X.

    Args:
        X (torch.Tensor): The input tensor of latent values with shape (batch_size, channels, height, width).
        r (float): Scaling factor that determines how far outside of the standard deviation a latent value needs to be to be penalized.

    Returns:
        torch.Tensor: The outlier penalty loss value.
    """
    # Compute mean and standard deviation of X over height and width dimensions (H, W)
    mean_X = X.mean(dim=(2, 3), keepdim=True)
    std_X = X.std(dim=(2, 3), keepdim=True)
    
    # Compute the penalty for each latent value
    penalty = torch.norm(X - mean_X, dim=1, keepdim=True) - r * std_X
    
    # Apply the max(., 0) operation
    penalty = torch.maximum(penalty, torch.zeros_like(penalty))
    
    # Average over height and width dimensions
    loss = penalty.mean(dim=(2, 3))
    
    return loss.mean()

def get_vgg16_feature_extractor(layers):
    from torchvision.models import vgg16, VGG16_Weights
    from torchvision.models.feature_extraction import create_feature_extractor
    m = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    return_nodes = [f"features.{l}" for l in layers]
    return create_feature_extractor(m, return_nodes)

class VGGLoss(nn.Module):
    """The feature reconstruction loss in Perceptual Losses for Real-Time Style Transfer and Super-Resolution (https://arxiv.org/abs/1603.08155)."""
    
    def __init__(self, layers=[3, 8, 15, 22]):
        super().__init__()
        self.feature_extractor = get_vgg16_feature_extractor(layers).eval()
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, x, y):
        x = self.feature_extractor(x)
        y = self.feature_extractor(y)
        loss = 0
        for k in x.keys():
            loss += torch.nn.functional.l1_loss(x[k], y[k], reduction="mean")
        return loss / len(x.keys())
    
def gaussian(x, sigma=1.0):
    return torch.exp(-(x**2) / (2*(sigma**2)))

def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, device=None):
    """Construct the convolution kernel for a gaussian blur

    See https://en.wikipedia.org/wiki/Gaussian_blur for a definition.
    Overall I first generate a NxNx2 matrix of indices, and then use those to
    calculate the gaussian function on each element. The two dimensional
    Gaussian function is then the product along axis=2.
    Also, in_channels == out_channels == n_channels
    """
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    
    # Create the grid using PyTorch.
    grid = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size)), dim=-1).float().to(device) - size // 2
    kernel = torch.prod(gaussian(grid, sigma), dim=-1)
    kernel /= torch.sum(kernel)

    # repeat same kernel for all pictures and all channels
    # Also, conv weight should be (out_channels, in_channels/groups, h, w)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions.
    kernel = kernel.repeat(n_channels, 1, 1, 1)  # Repeat the kernel to match the channel count.
    return kernel


def blur_images(images, kernel):
    """Convolve the gaussian kernel with the given stack of images"""
    _, n_channels, _, _ = images.shape
    _, _, kw, kh = kernel.shape
    imgs_padded = F.pad(images, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
    return F.conv2d(imgs_padded, kernel, groups=n_channels)


def laplacian_pyramid(images, kernel, max_levels=5):
    """Laplacian pyramid of each image

    https://en.wikipedia.org/wiki/Pyramid_(image_processing)#Laplacian_pyramid
    """
    current = images
    pyramid = []

    for level in range(max_levels):
        filtered = blur_images(current, kernel)
        diff = current - filtered
        pyramid.append(diff)
        current = F.avg_pool2d(filtered, 2)
    pyramid.append(current)
    return pyramid


class LapLoss(nn.Module):
    def __init__(self, max_levels=5, kernel_size=5, sigma=1.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._gauss_kernel = None

    def forward(self, output, target):
        if (self._gauss_kernel is None
                or self._gauss_kernel.shape[1] != output.shape[1]):
            self._gauss_kernel = build_gauss_kernel(
                n_channels=output.shape[1],
                device=output.device)
        output_pyramid = laplacian_pyramid(
            output, self._gauss_kernel, max_levels=self.max_levels)
        target_pyramid = laplacian_pyramid(
            target, self._gauss_kernel, max_levels=self.max_levels)
        diff_levels = [F.l1_loss(o, t)
                        for o, t in zip(output_pyramid, target_pyramid)]
        loss = sum([2**(-2*j) * diff_levels[j]
                    for j in range(self.max_levels)])
        return loss
    
def SSIM_(x, y, C1=1e-4, C2=9e-4, kernel_size=3, stride=1):
    """
    Structural SIMilarity (SSIM) distance between two images.

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    C1,C2 : float
        SSIM parameters
    kernel_size,stride : int
        Convolutional parameters

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM distance
    """
    pool2d = nn.AvgPool2d(kernel_size, stride=stride)
    refl = nn.ReflectionPad2d(1)

    x, y = refl(x), refl(y)
    mu_x = pool2d(x)
    mu_y = pool2d(y)

    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = pool2d(x.pow(2)) - mu_x_sq
    sigma_y = pool2d(y.pow(2)) - mu_y_sq
    sigma_xy = pool2d(x * y) - mu_x_mu_y
    v1 = 2 * sigma_xy + C2
    v2 = sigma_x + sigma_y + C2

    ssim_n = (2 * mu_x_mu_y + C1) * v1
    ssim_d = (mu_x_sq + mu_y_sq + C1) * v2
    ssim = ssim_n / ssim_d

    return ssim

def SSIM(x, y, kernel_size=3):
    """
    Calculates the SSIM (Structural SIMilarity) loss

    Parameters
    ----------
    x,y : torch.Tensor [B,3,H,W]
        Input images
    kernel_size : int
        Convolutional parameter

    Returns
    -------
    ssim : torch.Tensor [1]
        SSIM loss
    """
    ssim_value = SSIM_(x, y, kernel_size=kernel_size)
    return torch.clamp((1. - ssim_value) / 2., 0., 1.)

def calc_photometric_loss(t_est, images, ssim_loss_weight=0.85, clip_loss=0):
    """
    Calculates the photometric loss (L1 + SSIM)
    Parameters
    ----------
    t_est : list of torch.Tensor [B,3,H,W]
        List of warped reference images in multiple scales
    images : list of torch.Tensor [B,3,H,W]
        List of original images in multiple scales

    Returns
    -------
    photometric_loss : torch.Tensor [1]
        Photometric loss
    """
    # L1 loss
    l1_loss = [torch.abs(t_est[i] - images[i])
                for i in range(len(t_est))]
    # SSIM loss
    if ssim_loss_weight > 0.0:
        ssim_loss = [SSIM(t_est[i], images[i], kernel_size=3)
                        for i in range(len(t_est))]
        # Weighted Sum: alpha * ssim + (1 - alpha) * l1
        photometric_loss = [ssim_loss_weight * ssim_loss[i].mean(1, True) +
                            (1 - ssim_loss_weight) * l1_loss[i].mean(1, True)
                            for i in range(len(t_est))]
    else:
        photometric_loss = l1_loss
    # Clip loss
    if clip_loss > 0.0:
        for i in range(len(photometric_loss)):
            mean, std = photometric_loss[i].mean(), photometric_loss[i].std()
            photometric_loss[i] = torch.clamp(
                photometric_loss[i], max=float(mean + clip_loss * std))
    # Return total photometric loss
    return photometric_loss

def reduce_photometric_loss(photometric_losses, photometric_reduce_op='min'):
    """
    Combine the photometric loss from all context images

    Parameters
    ----------
    photometric_losses : list of torch.Tensor [B,3,H,W]
        Pixel-wise photometric losses from the entire context

    Returns
    -------
    photometric_loss : torch.Tensor [1]
        Reduced photometric loss
    """
    # Reduce function
    def reduce_function(losses):
        if photometric_reduce_op == 'mean':
            return sum([l.mean() for l in losses]) / len(losses)
        elif photometric_reduce_op == 'min':
            return torch.cat(losses, 1).min(1, True)[0].mean()
        else:
            raise NotImplementedError(
                'Unknown photometric_reduce_op: {}'.format(photometric_reduce_op))
    # Reduce photometric loss
    photometric_loss = sum([reduce_function(photometric_losses[i])
                            for i in range(len(photometric_losses))]) / len(photometric_losses)
    return photometric_loss

def final_cross_entropy(logits: Tensor, labels: Tensor, ignore_index: int = 255) -> Tensor:
    """Supervise final semantic predictions; all-ignored batches return zero."""
    if logits.shape[-2:] != labels.shape[-2:]:
        logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    loss_map = F.cross_entropy(logits, labels, reduction='none', ignore_index=ignore_index)
    valid = labels.ne(ignore_index)
    return (loss_map * valid).sum() / valid.sum().clamp_min(1)


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: tuple = (1, 0, 0)) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights) if w != 0])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: tuple = (1, 0)) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights) if w != 0])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: tuple = (1, 0, 0)):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights) if w != 0])
        return self._forward(preds, targets)


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights)


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)