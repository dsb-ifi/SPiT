import torch
import torch.nn.functional as F

from typing import Union, Sequence

def unravel_index(
    indices:torch.Tensor, shape:Union[Sequence[int], torch.Tensor]
) -> torch.Tensor:
    '''Converts a tensor of flat indices into a tensor of coordinate vectors.

    Args:
        index (torch.Tensor): Indices to unravel.
        shape (tuple[int]): Shape of tensor.

    Returns:
        torch.Tensor: Tensor (long) of unraveled indices.
    '''
    try:
        shape = indices.new_tensor(torch.Size(shape))[:,None] # type: ignore
    except Exception:
        pass
    shape = F.pad(shape, (0,0,0,1), value=1)                  # type: ignore
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()
    return torch.div(indices[None], coefs, rounding_mode='trunc') % shape[:-1]


def fast_uidx_1d(ar:torch.Tensor) -> torch.Tensor:
    '''Pretty fast unique index calculation for 1d tensors.

    Args:
        ar (torch.Tensor): Tensor to compute unique indices for.

    Returns:
        torch.Tensor: Tensor (long) of indices.
    '''
    assert ar.ndim == 1, f'Need dim of 1, got: {ar.ndim}!'
    perm = ar.argsort()
    aux = ar[perm]
    mask = ar.new_zeros(aux.shape[0], dtype=torch.bool)
    mask[:1] = True
    mask[1:] = aux[1:] != aux[:-1]
    return perm[mask]


def fast_uidx_long2d(ar:torch.Tensor) -> torch.Tensor:
    '''Pretty fast unique index calculation for 2d long tensors (row wise).

    Args:
        ar (torch.Tensor): Tensor to compute unique indices for.

    Returns:
        torch.Tensor: Tensor (long) of indices.
    '''
    assert ar.ndim == 2, f'Need dim of 2, got: {ar.ndim}!'
    m = ar.max() + 1
    r, c = ar
    cons = r*m + c
    return fast_uidx_1d(cons)


def lexsort(*tensors:torch.Tensor) -> torch.Tensor:
    '''Lexicographical sort of multidimensional tensor.

    Args:
       src (torch.Tensor): Input tensor.
        dim (int): Dimension to sort over, defaults to -1.

    Returns:
        torch.Tensor: Sorting indices for multidimensional tensor.
    '''
    numel = tensors[0].numel()
    assert all([t.ndim == 1 for t in tensors])
    assert all([t.numel() == numel for t in tensors[1:]])
    idx = tensors[0].argsort(dim=0, stable=True)
    for k in tensors[1:]:
        idx = idx.gather(0, k.gather(0, idx).argsort(dim=0, stable=True))
    return idx