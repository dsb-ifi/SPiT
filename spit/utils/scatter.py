import torch

def scatter_mean_2d(src:torch.Tensor, idx:torch.Tensor) -> torch.Tensor:
    '''Computes scatter mean with 2d source and 1d index over first dimension.

    Args:
        src (torch.Tensor): Source tensor.
        idx (torch.Tensor): Index tensor.

    Returns:
        torch.Tensor: Output tensor.
    '''
    assert src.ndim == 2
    assert len(src) == len(idx)
    if idx.ndim == 1:
        idx = idx.unsqueeze(1).expand(*src.shape)
    out = src.new_empty(idx.max()+1, src.shape[1]) # type: ignore
    return out.scatter_reduce_(0, idx, src, 'mean', include_self=False)


def scatter_add_1d(src:torch.Tensor, idx:torch.Tensor, n:int) -> torch.Tensor:
    '''Computes scatter add with 1d source and 1d index.

    Args:
        src (torch.Tensor): Source tensor.
        idx (torch.Tensor): Index tensor.
        n (int): No. outputs.

    Returns:
        torch.Tensor: Output tensor.
    '''
    assert src.ndim == 1
    assert len(src) == len(idx)
    out = src.new_zeros(n)
    return out.scatter_add_(0, idx, src)


def scatter_range_2d(src:torch.Tensor, idx:torch.Tensor) -> torch.Tensor:
    '''Computes scattered range (max-min) with 2d source and 1d index.

    Args:
        src (torch.Tensor): Source tensor.
        idx (torch.Tensor): Index tensor.

    Returns
        torch.Tensor: Output tensor.
    '''
    if idx.ndim == 1:
        idx = idx.unsqueeze(1).expand(*src.shape)
    mx = src.new_empty(idx.max()+1, src.shape[1]) # type: ignore
    mn = src.new_empty(idx.max()+1, src.shape[1]) # type: ignore
    mx.scatter_reduce_(0, idx, src, 'amax', include_self=False)
    mn.scatter_reduce_(0, idx, src, 'amin', include_self=False)
    return mx - mn
    

def scatter_cov_2d(src:torch.Tensor, idx:torch.Tensor) -> torch.Tensor:
    '''Scatter covariance reduction.

    NOTE: Runs two passes.

    Args:
        src (torch.Tensor): Source tensor.
        idx (torch.Tensor): Index tensor.

    Returns:
        torch.Tensor: Output tensor.
    '''
    d = src.shape[-1]
    mu = scatter_mean_2d(src, idx)
    diff = (src - mu[idx])
    return scatter_mean_2d(
        (diff.unsqueeze(-1) @ diff.unsqueeze(-2)).view(-1,d**2), 
        idx,
    )
