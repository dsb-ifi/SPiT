import torch
import cupy
import os
from typing import Optional


__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)

with open(os.path.join(__location__, 'cuda', 'cossimargmax.cu'), 'r') as csam_file:
    _argmax_cosine_kernel_code = csam_file.read()

_argmax_cosine_kernel = cupy.RawKernel(
    _argmax_cosine_kernel_code, 'argmax_cosine_kernel'
)

_argmax_cosine_kernel_bbox = cupy.RawKernel(
    _argmax_cosine_kernel_code, 'argmax_cosine_kernel_bbox'
)

def argmax_cosine_gpu(
    vfeat:torch.Tensor, edges:torch.Tensor, sizes:torch.Tensor, nnz:int, 
    tpb:int=1024, lvl:Optional[int]=None, bbox:Optional[torch.Tensor]=None,
    cmix:float=0.1
):
    '''Computes argmax over cosine similarity in a graph using custom CUDA kernel.
    
    Secures parallel execution by packing similarity scores and argmax indices
    in single unsigned long long where the highest bits encode the similarity.
    This ensures that we can use atomicMax calls to update the values on both
    edges, which eliminates race conditions. Since we do not have to initialise
    multiple potentially long arrays, this saves on memory consumption, while 
    ensuring good performance. Faster than reduction, sacrificing some precision
    in the similarities.

    The packing is done by encoding the similarities as uint16. We bitshift by 48
    and then the indices of the array is `or`'ed into the remaining bits. After applying
    atomicMax, the result is `and`'ed with a 48-bit value to retrieve the indices. 

    Args:
        vfeat (torch.Tensor): Vertex features.
        edges (torch.Tensor): Edge indices.
        sizes (torch.Tensor): Current node sizes.
        tpb (int, optional): Threads per block.
        lvl (int, optional): Current level.
    '''
    dtype = vfeat.dtype
    u, v = edges[0].contiguous(), edges[1].contiguous()
    vfeat = vfeat.contiguous()
    sfl = sizes.contiguous().to(dtype=dtype)
    if lvl is None:
        mu = sfl.mean()[None] * 2
    else:
        # Cheaper to use expected value than compute mean size
        # but less adaptive to current samples.
        mu = sfl.new_tensor([4**(lvl-1)]) * 2
    std = sfl.std().clip(min=1e-6)[None]    
    cmixptr = sfl.new_tensor([cmix])
    m = len(u)
    d = vfeat.shape[-1]
    packed = edges.new_zeros(nnz)
    bpg = (m + tpb - 1) // tpb
    
    _todl = cupy.from_dlpack
    with cupy.cuda.Device(vfeat.device.index) as cpdev:
        if dtype == torch.float:
            if bbox is None:
                kernel = _argmax_cosine_kernel
            else:
                kernel = _argmax_cosine_kernel_bbox
        else:
            raise TypeError(f'No support for dtype:{dtype}')

        if bbox is None:
            kernel(
                (bpg,), (tpb,), (
                    m, d, nnz, 
                    _todl(vfeat), _todl(u), _todl(v), 
                    _todl(packed), _todl(mu), 
                    _todl(std), _todl(sfl)
                )
            )
        else:
            ymin, xmin, ymax, xmax = bbox.to(dtype=dtype)
            kernel(
                (bpg,), (tpb,), (
                    m, d, nnz, 
                    _todl(vfeat), _todl(u), _todl(v), 
                    _todl(packed), _todl(mu), _todl(std), _todl(cmixptr),
                    _todl(sfl), _todl(ymin), _todl(xmin), _todl(ymax), _todl(xmax)
                )
            )
        
    return packed & 0xFFFFFFFFFFFF


def packed_scatter_argmax(src:torch.Tensor, idx:torch.Tensor, n:int) -> torch.Tensor:
    '''Computes scatter argmax with 1d source and 1d index.
    
    Uses packing, i.e., reduced precision over the tensors to retrieve the argmax.
    Assumes inputs in range [-1, 1]. Could be generalized by scaling but, meh.
    Packs the src tensor as a virtual uint16, and bitshifts by 47, avoiding
    the sign bit of the int64. The indices of the array is or'ed into the remaining
    bits. The atomicMax operation takes the max over the src. The result is then 
    and'ed with the 47-bit representation to retrieve the indices. 

    NOTE: This is almost how the CUDA kernel works, except by considering the inputs
          as unsigned long longs, we squeeze a little more headroom for large indices.

    Args:
        src (torch.Tensor): Source tensor in range [-1, 1].
        idx (torch.Tensor): Index tensor.
        n (int): No. outputs.

    Returns:
        torch.Tensor: Output tensor.
    '''
    assert src.ndim == 1
    assert len(src) == len(idx)
    assert (len(src) & 0x7FFF800000000000) == 0

    shorts = (src.clip(-1, 1).add(1).div(2) * (2**16 - 1)).long()
    packed = (shorts << 47) | torch.arange(len(src), device=src.device)
    out = packed.new_zeros(n)
    out.scatter_reduce_(0, idx, packed, 'amax', include_self=False)
    return out & 0x7FFFFFFFFFFF


def argmax_cosine_pytorch(vfeat, edges, sizes, nnz, lvl=None):
    '''Computes argmax over cosine similarity in a graph using pytorch.
    
    This is device agnostic, but requires much higher memory overhead.

    Args:
        V (torch.Tensor): Vertex features.
        E (torch.Tensor): Edge indices.
        s (torch.Tensor): Current node sizes.
        m (float): Mean of current sizes.
    '''
    u, v = edges
    sfl = sizes.float()
    if lvl is None:
        mu = sfl.mean()
    else:
        mu = sfl.new_tensor(4**(lvl-1))
    stdwt = (sfl - mu) / sfl.std().clip(min=1e-6)
    weight = torch.where(u == v, stdwt[u].clip(-.75, .75), 1.0)
    sim = torch.cosine_similarity(vfeat[u], vfeat[v]) * weight
    udir, vdir = torch.cat([edges, edges.flip(0)], 1)
    simdir = sim[None,:].expand(2,-1).reshape(-1)
    argmax_idx = packed_scatter_argmax(simdir, udir, nnz)
    return vdir[argmax_idx]


def cosine_similarity_argmax(vfeat, edges, sizes, nnz, force_pt=False, lvl:Optional[int]=None, bbox:Optional[torch.Tensor]=None, tpb=1024):
    '''Computes argmax over cosine similarity in a graph using pytorch.
    
    Checks if tensors are located on CPU or GPU, preferring the custom CUDA 
    implementation to PyTorch for improved memory footprint.

    Args:
        vfeat (torch.Tensor): Vertex features.
        edges (torch.Tensor): Edge indices.
        sizes (torch.Tensor): Current node sizes.
        nnz (float): Number of vertices.
    '''
    assert vfeat.device == edges.device
    assert vfeat.device == sizes.device    
    if vfeat.device.type == 'cpu' or force_pt:
        return argmax_cosine_pytorch(vfeat, edges, sizes, nnz, lvl=lvl)
    return argmax_cosine_gpu(vfeat, edges, sizes, nnz, tpb=tpb, lvl=lvl, bbox=bbox)
