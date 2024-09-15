import torch

from typing import Optional
from ..utils.indexing import fast_uidx_1d
from ..utils.scatter import scatter_mean_2d, scatter_cov_2d


def voronoi(img: torch.Tensor, num_cells:int) -> torch.Tensor:
    '''Constructs a Voronoi partition.

    NOTE: In the paper, the RViT models were trained
          using pregenerated Voronoi partitions, as on-line
          computations are quite memory intensive.

    Args:
        img (torch.Tensor): An image.
        num_cells (int): Desired number of cells for the partition.

    Returns:
        Voronoi partitioning.
    '''
    B,_,H,W = img.shape
    N = B*H*W
    C = num_cells
    dev = img.device
    shape = torch.tensor([B,H,W,1], device=dev)[:,None]
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()
    byx = torch.div(torch.arange(N, device=dev)[None], coefs, rounding_mode='trunc') % shape[:-1]
    y, x = byx[1:] / torch.tensor([H,W], device=byx.device)[:,None]
    b = byx[0]
    sy,sx = torch.rand(2,B,C,device=byx.device)
    std = C/(H*W)**.5
    
    # One liner gaussian kernels
    def gauss1d(x): return x.div(std).pow_(2).neg_().exp_()
    def gauss2d(x, y): return (gauss1d(x) + gauss1d(y)) / 2

    out = gauss2d(y[:,None] - sy[b], x[:,None] - sx[b]).argmax(-1).view(B,H,W)
    return out


def chunked_voronoi(img: torch.Tensor, num_cells:int, chunks:int=16) -> torch.Tensor:
    '''Uses chunking to lower memory overhead of Voronoi.

    NOTE: In the paper, the RViT models were trained
          using pregenerated Voronoi partitions, as on-line
          computations are quite memory intensive.

    Args:
        img (torch.Tensor): An image.
        num_cells (int): Desired number of cells for the partition.
        chunks (int): Number of chunks to use.

    Returns:
        Voronoi partitioning.
    '''
    B, _, H, W = img.shape
    outs = [] 
    cums = 0
    for c, chunk in enumerate(img.chunk(chunks, 0)):
        vor = voronoi(chunk, num_cells) + cums
        cums = vor.max().item()
        outs.append(vor)

    out = torch.cat(outs, 0)
    return out.view(-1).unique(return_inverse=True)[1].view(B,H,W)


def _init_sobol_spatial_centroids(
    img:torch.Tensor, num_cells:int
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:
    B, _, H, W = img.shape
    device = img.device
    engine = torch.quasirandom.SobolEngine(2, True)
    eps = 1e-7
    centroids = (
        engine.draw(B*num_cells).clip(0,1-1e-7) * 
        torch.tensor([[H,W]])
    ).round().long().to(device)
    init_index = torch.arange(B, device=device).repeat_interleave(num_cells)
    sizes = torch.full((B,), num_cells, dtype=torch.long, device=device)
    return centroids, init_index, sizes

def _pcatree(
    points:torch.Tensor, cur_idx:torch.Tensor, 
    cur_sizes:Optional[torch.Tensor]=None, steps:Optional[int]=None
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    if cur_sizes is None:
        cur_sizes = cur_idx.bincount()
    if steps is None:
        num_cells = points.shape[0] // (cur_idx.max().item() + 1)
        steps = int(torch.log2(torch.tensor(num_cells)).ceil().long().item())

    all_weights = []
    all_mus = []
    indices = [cur_idx]

    for _ in range(steps):
        cov = scatter_cov_2d(points, cur_idx).view(-1,2,2)
        mu = scatter_mean_2d(points, cur_idx)
        clus_rng = torch.arange(len(cov))
        
        eigval, eigvec = torch.linalg.eigh(cov)
        weights = eigvec[clus_rng, eigval.argmax(-1)]
        cpoints = points - mu[cur_idx]
        dot_products = (cpoints * weights[cur_idx]).sum(-1)
        split = dot_products >= 0
        cur_idx = 2*cur_idx + split
        all_weights.append(weights)
        all_mus.append(mu)
        indices.append(cur_idx)

    return indices, all_weights, all_mus


def fast_pseudo_voronoi(
    img:torch.Tensor, num_cells:int
) -> torch.Tensor:
    '''Computes a fast pseudo Voronoi tesselation.
    
    NOTE: This method uses PCA Trees first proposed by Sproull, 1991.
          While not strictly Voronoi tesselations, they are sufficiently
          similar, and samples faster O(n log n).
          https://doi.org/10.1007/BF01759061

    Args:
        img (torch.Tensor): Input image of shape B,C.H,W.
        num_cells (int): Number of cells to compute in tree. Should ideally be a power of 2.

    Returns:
        torch.Tensor: Pseudo Voronoi partitioning.
    '''
    B, _, H, W = img.shape
    device = img.device
    points, cur_idx, sizes = _init_sobol_spatial_centroids(img, num_cells)
    points = points.float()
    
    _, weights, mus = _pcatree(points, cur_idx, sizes)
    
    byx = torch.stack(torch.meshgrid(
        torch.arange(B, device=device), 
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    ), 0).view(3,-1).mT
    
    idx, cur_idx, yx = torch.arange(B*H*W), byx[:,0], byx[:,1:]
    yx = yx.float()
    steps = len(weights)

    for step in range(steps):
        split = (
            (yx - mus[step][cur_idx]) * weights[step][cur_idx]
        ).sum(-1) >= 0
        cur_idx = 2*cur_idx + split

    return cur_idx.unique(return_inverse=True)[1].view(B,H,W)



