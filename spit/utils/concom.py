import torch
import os
import cupy
from warnings import warn
from scipy.sparse import coo_matrix as cpu_coo_matrix
from scipy.sparse.csgraph import connected_components as cpu_concom
from typing import Optional


__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)

with open(os.path.join(__location__, 'cuda', 'concom.cu'), 'r') as cc_file:
    _dpcc_roots_kernel_code = cc_file.read()

_dpcc_roots_kernel = cupy.RawKernel(
    _dpcc_roots_kernel_code, 'dpcc_roots_kernel'
)

def dpcc_recursive(
    n:int, u:torch.Tensor, v:torch.Tensor, labels:torch.Tensor,
    it:int, maxit:int=50, tpb:int=128, bpg:Optional[int]=None
):
    '''Recursive loop for parallel connected components for CUDA.

    Args:
        n (int): Number of vertices in graph.
        u (torch.Tensor): Source edges.
        v (torch.Tensor): Target edges.
        labels (torch.Tensor): Labels of nodes.
        it (int): Current iteration.
        maxit (int, optional): Max number of iterations.
        tpb (int, optional): Threads per block.
        bpg (int, optional): Blocks per grid (def: get from n, tpg).
    '''
    assert labels.device == u.device
    assert labels.device == v.device
    
    # Get current number of edges
    m = len(u)

    # Calculate default blocks per grid
    if bpg is None:
        bpg = (n + (tpb - 1)) // tpb

    # Convergence, end recursion
    if m == 0:
        return

    # Maximum iterations
    if it > maxit:
        msg = f'DPCC recursion limit - curit: {it} > maxit: {maxit}.'
        warn(msg, RuntimeWarning)
        return

    # Low->High vs. High->Low idxcount
    l2h = (u < v).sum()
    h2l = m - l2h

    # Pick largest for maximum graph reduction
    if l2h >= h2l:
        mask = u < v
    else:
        mask = u > v

    # Contract labels
    labels[u[mask]] = v[mask]

    # Compute roots
    with cupy.cuda.Device(labels.device.index):
        _dpcc_roots_kernel((bpg,), (tpb,), (n, cupy.from_dlpack(labels)))
    
    # Compute new edges
    mask = labels[u] != labels[v]
    uprime = labels[u[mask]]
    vprime = labels[v[mask]]
    
    # Recurse
    dpcc_recursive(n, uprime, vprime, labels, it+1, maxit, tpb, bpg)

    
def cc_gpu(src:torch.Tensor, tgt:torch.Tensor, n:int, tpb:int=128) -> torch.Tensor:
    '''Parallel connected components algorithm on CUDA.

    Args:
        src (int): Source edges.
        tgt (int): Target edges.
        n (int): Number of vertices in graph.

    Returns:
        torch.Tensor: Connected components of graph.
    '''
    # Init labels
    device = src.device
    labels = torch.arange(n, device=device)

    # Connected Components
    dpcc_recursive(n, src, tgt, labels, 0, tpb=tpb)

    # Return unique inverse
    return labels.unique(return_inverse=True)[1]


def cc_cpu(src:torch.Tensor, tgt:torch.Tensor, n:int) -> torch.Tensor:
    '''Computes connected components using SciPy / CPU

    Args:
        src (int): Source edges.
        tgt (int): Target edges.
        n (int): Number of vertices in graph.

    Returns:
        torch.Tensor: Connected components of graph.
    '''
    ones = torch.ones_like(src, device='cpu').numpy()
    edges = (src.numpy(), tgt.numpy())
    csr = cpu_coo_matrix((ones, edges), shape=(n,n)).tocsr()
    return src.new_tensor(cpu_concom(csr)[1])


def connected_components(src:torch.Tensor, tgt:torch.Tensor, n:int, tpb=1024) -> torch.Tensor:
    '''Connected components algorithm (device agnostic).

    Args:
        src (int): Source edges.
        tgt (int): Target edges.
        n (int): Number of vertices in graph.

    Returns:
        torch.Tensor: Connected components of graph.
    '''
    assert src.shape == tgt.shape
    if src.device.type == 'cpu':
        return cc_cpu(src, tgt, n)
    return cc_gpu(src, tgt, n, tpb=tpb)