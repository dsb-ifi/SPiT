import torch
import torch.nn.functional as F

_in1k_mean = torch.tensor([0.485, 0.456, 0.406])
_in1k_std = torch.tensor([0.229, 0.224, 0.225])

def _in1k_norm(tensor, dim=-1):
    shape = [1] * tensor.ndim
    shape[dim] = -1
    mean = _in1k_mean.view(shape).to(tensor.device)
    std = _in1k_std.reshape(shape).to(tensor.device)
    return (tensor - mean) / std

def _in1k_unnorm(tensor, dim=-1):
    shape = [1] * tensor.ndim
    shape[dim] = -1
    mean = _in1k_mean.view(shape).to(tensor.device)
    std = _in1k_std.reshape(shape).to(tensor.device)
    return tensor * std + mean

def kuma_contrast1(x:torch.Tensor, mu:float, lambda_:float):
    '''Contrast adjustment with Kumaraswamy in range -1, 1.

    Args:
        x (torch.Tensor): Feature tensor.
        mu (float): Mean for contrast.
        lambda_ (float): Shape parameter for contrast.

    Returns:
        torch.Tensor: Adjusted features.
    '''
    x = x.clip(0,1)
    m, a = x.new_tensor(mu).clip_(0, 1), x.new_tensor(lambda_).clip_(0)
    b = -(x.new_tensor(2)).log_() / (1-m**a).log_()
    return 1 - (1 - x**a)**b


def kuma_contrast1_(x:torch.Tensor, mu:float, lambda_:float):
    '''Contrast adjustment with Kumaraswamy in range -1, 1.

    Args:
        x (torch.Tensor): Feature tensor.
        mu (float): Mean for contrast.
        lambda_ (float): Shape parameter for contrast.

    Returns:
        torch.Tensor: Adjusted features.
    '''
    x.clip_(0,1)
    m, a = x.new_tensor(mu).clip_(0, 1), x.new_tensor(lambda_).clip_(0)
    b = -(x.new_tensor(2)).log_() / (1-m**a).log_()
    return x.pow_(a).mul_(-1).add_(1).pow_(b).mul_(-1).add_(1)


def asinh_contrast(features:torch.Tensor, lambda_:float) -> torch.Tensor:
    '''Contrast adjustment with Arcsinh.

    Args:
        features (torch.Tensor): Feature tensor.
        lambda_ (float): Multiplier for contrast.

    Returns:
        torch.Tensor: Adjusted features.
    '''
    if lambda_ == 0:
        return features
    tmul = features.new_tensor(lambda_)
    m, d = tmul, torch.arcsinh(tmul)
    if lambda_ > 0:
        return features.mul(m).arcsinh().div(d)
    return features.mul(d).sinh().div(m)


def asinh_contrast_(features:torch.Tensor, lambda_:float) -> torch.Tensor:
    '''In-place contrast adjustment with Arcsinh.

    Args:
        features (torch.Tensor): Feature tensor.
        lambda_ (float): Multiplier for contrast.

    Returns:
        torch.Tensor: Adjusted features.
    '''
    if lambda_ == 0:
        return features
    tmul = features.new_tensor(lambda_)
    m, d = tmul, torch.arcsinh(tmul)
    if lambda_ > 0:
        return features.mul_(m).arcsinh_().div_(d)
    return features.mul_(d).sinh_().div_(m)


def pthroot_contrast(features:torch.Tensor, lambda_:float) -> torch.Tensor:
    '''Hard contrast adjustment with pth-root.

    Args:
        features (torch.Tensor): Feature tensor.
        lambda_ (float): Pth root for contrast.

    Returns:
        torch.Tensor: Adjusted features.
    '''
    mul = features.sign().div_(features.new_tensor(1.0).exp_().sub_(1).pow_(1/lambda_))
    return features.abs().exp_().sub_(1).pow_(1/lambda_).mul_(mul)


def pthroot_contrast_(features:torch.Tensor, lambda_:float) -> torch.Tensor:
    '''In-place hard contrast adjustment with pth-root.

    Args:
        features (torch.Tensor): Feature tensor.
        lambda_ (float): Pth root for contrast.

    Returns:
        torch.Tensor: Adjusted features.
    '''
    mul = features.sign().div_(features.new_tensor(1.0).exp_().sub_(1).pow_(1/lambda_))
    features.abs_().exp_().sub_(1).pow_(1/lambda_).mul_(mul)
    return features


def scharr_features(img:torch.Tensor, lambda_:float) -> torch.Tensor:
    '''Computes constrast enchanced Scharr featrues of an image.

    Args:
        img (torch.Tensor): Image tensor.
        contrast (float): Multiplier for contrast.

    Returns:
        torch.Tensor: Adjusted features.
    '''
    img = img.mean(1, keepdim=True)
    kernel = img.new_tensor([[[[-3.,-10,-3.],[0.,0.,0.],[3.,10,3.]]]])
    kernel = torch.cat([kernel, kernel.mT], dim=0)
    out = F.conv2d(
        F.pad(img, 4*[1], mode='replicate'), 
        kernel, 
        stride=1
    ).div_(16)
    return asinh_contrast_(out, lambda_)


def adjust_saturation(rgb:torch.Tensor, mul:float):
    '''Adjusts saturation via interpolation / extrapolation.

    Args:
        rgb (torch.Tensor): An input tensor of shape (..., 3) representing the RGB values of an image.
        mul (float): Saturation adjustment factor. A value of 1.0 will keep the saturation unchanged.

    Returns:
        torch.Tensor: A tensor of the same shape as the input, with adjusted saturation.
    """    
    '''
    weights = rgb.new_tensor([0.299, 0.587, 0.114])
    grayscale = torch.matmul(rgb, weights).unsqueeze(dim=-1).expand_as(rgb).to(dtype=rgb.dtype)
    return torch.lerp(grayscale, rgb, mul).clip(0,1)


def peronamalik1(img, niter=5, kappa=0.0275, gamma=0.275):
    """Anisotropic diffusion.
    
    Perona-Malik anisotropic diffusion type 1, which favours high contrast 
    edges over low contrast ones.
    
    `kappa` controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.
           
    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Args:
        img (torch.Tensor): input image
        niter (int): number of iterations
        kappa (float): conduction coefficient.
        gamma (float): controls speed of diffusion (generally max 0.25)
    
    Returns:
    Diffused image.    
    """
    
    deltaS, deltaE = img.new_zeros(2, *img.shape)
    
    for _ in range(niter):
        deltaS[...,:-1,:] = torch.diff(img, dim=-2)
        deltaE[...,:,:-1] = torch.diff(img, dim=-1)

        gS = torch.exp(-(deltaS/kappa)**2.)
        gE = torch.exp(-(deltaE/kappa)**2.)
        
        S, E = gS*deltaS, gE*deltaE

        S[...,1:,:] = S.diff(dim=-2)
        E[...,:,1:] = E.diff(dim=-1)
        img = img + gamma*(S+E)
    
    return img


def rgb_to_ycbcr(feat: torch.Tensor, dim=-1) -> torch.Tensor:
    r"""Convert RGB features to YCbCr.

    Args:
        feat (torch.Tensor): Pixels to be converted YCbCr.

    Returns:
        torch.Tensor: YCbCr converted features.
    """    
    r,g,b = feat.unbind(dim)
    y = 0.299 * r + 0.587 * g + 0.114 * b
    delta = 0.5
    cb = (b - y) * 0.564 + delta
    cr = (r - y) * 0.713 + delta
    return torch.stack([y, cb, cr], dim)


def apply_color_transform(
    colfeat:torch.Tensor, shape:tuple[int,...], lambda_col:float,
) -> torch.Tensor:
    ''' Applies anisotropic diffusion and color enhancement.

    Args:
        colfeat (torch.Tensor): Color features.
        shape (tuple[int]): Image shape.
        lambda_col (int): Color contrast to use.
    '''
    b, _ , h, w = shape
    c = colfeat.shape[-1]
    f = adjust_saturation(colfeat.add(1).div_(2), 2.718)

    f = rgb_to_ycbcr(f, -1).mul_(2).sub_(1)
    asinh_contrast_(f, lambda_col)
    f = peronamalik1(
        f.view(b, h, w, c).permute(0,3,1,2),
        4, 
        0.1,
        0.5
    ).permute(0,2,3,1).view(-1, c).clip_(-1,1)
    return f
