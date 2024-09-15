import torch
import torch.nn.functional as F
import numpy, cupy
import os

try:
    import numba 

    @numba.jit(nopython=True, parallel=True)
    def _flatnorm_scatterhist_kernel_cpu( # type: ignore
        features, output, indices, bins, 
        sigma, num_pixels, num_features, num_bins
    ):
        for idx in numba.prange(num_pixels):
            output_idx = indices[idx]
            for feature in range(num_features):
                feat_val = features[idx][feature]

                for bin in range(num_bins):
                    bin_val = bins[bin]
                    diff = feat_val - bin_val
                    hist_val = numpy.exp(-0.5 * (diff / sigma) ** 2)
                    output[output_idx][feature * num_bins + bin] += hist_val

except ImportError:
    numba = None
    def _flatnorm_scatterhist_kernel_cpu(
        features, output, indices, bins, 
        sigma, num_pixels, num_features, num_bins
    ):
        raise NotImplementedError('Numba not installed, only CUDA available.')
        

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)

with open(os.path.join(__location__, 'cuda', 'scatterhist.cu'), 'r') as sch_file:
    _scatterhist_kernel_code = sch_file.read()

_flatnorm_scatterhist_kernel = cupy.RawKernel(
    _scatterhist_kernel_code, 'flatnorm_scatterhist_kernel'
)

_scatter_joint_hist_kernel = cupy.RawKernel(
    _scatterhist_kernel_code, 'scatter_joint_hist'
)

def scatter_hist(
    mapping:torch.Tensor, features:torch.Tensor, num_bins:int, 
    sigma:float=0.025, low:float=-1, high:float=1, tpb=128
) -> torch.Tensor:
    '''Scattered histogram computation.
    
    Effectively computes KDE histograms with a Gaussian kernel using scatter operations.
    Args:
        mapping (torch.Tensor): Hierograph mapping, precomputed.
        features (torch.Tensor): Base pixel features.
        num_bins (int): Level to compute.
        low (int, optional): Min val of bins.
        high (int, optional): Max val of bins.
        tpb (int, optional): Threads per block for GPU. 
    '''
    device = features.device
    dtype = features.dtype
    delta = 1/num_bins
    bins = torch.linspace(low+delta, high-delta, num_bins, dtype=dtype, device=device)
    num_pixels, num_features = features.shape    
    output = features.new_zeros(
        mapping.max()+1,                                                    # type: ignore
        num_features*num_bins, 
        dtype=torch.float # Use float as output for portability with 6.1 devices.
    )
    
    if device.type == 'cpu':
        _tonp = lambda x: x.numpy()
        _flatnorm_scatterhist_kernel_cpu(
            _tonp(features),
            _tonp(output),
            _tonp(mapping),
            _tonp(bins),
            sigma,
            num_pixels,
            num_features,
            num_bins,
        )
        return output
    
    sigma = features.new_tensor([sigma])                                    # type: ignore
    bpg = (num_pixels + (tpb - 1)) // tpb
    _todl = cupy.from_dlpack
    with cupy.cuda.Device(device.index):
        if dtype == torch.half:
            raise NotImplementedError()
        elif dtype == torch.float:
            kernel = _flatnorm_scatterhist_kernel
        else:
            raise TypeError(f'No support for dtype:{dtype}')
        kernel(
            (bpg,), (tpb,), (
                _todl(features),
                _todl(output),
                _todl(mapping),
                _todl(bins),
                _todl(sigma),
                num_pixels,
                num_features,
                num_bins,
            )
        )
        
    return output.to(dtype=dtype)


def scatter_joint_hist(
    seg:torch.Tensor, feats:torch.Tensor, num_seg, num_bins, featcombs,
    sigma=0.025, low=-1, high=1,
    tpb=1024,
):
    '''Scattered histogram computation.
    
    Effectively computes KDE 2d histograms with a Gaussian kernel using scatter operations.
    Args:
        seg (torch.Tensor): Segmentation.
        feats (torch.Tensor): Base pixel features.
        num_seg (int): Number of superpixels.
        num_bins (int): Number of bins in each dimension.
        featcombs (list[tuple[int, int]]): Index of joint features for histogram.
        sigma (float): Bandwith of KDE kernel.
        low (int, optional): Min val of bins.
        high (int, optional): Max val of bins.
        tpb (int, optional): Threads per block for GPU. 

    Returns:
        Tensor: 2D KDE histogram of features.
    '''    
    n, feat_dim = feats.shape
    delta = 1/num_bins
    featcombs = seg.new_tensor(featcombs)
    num_feats = len(featcombs)

    assert n == len(seg)
    assert featcombs.max() < feat_dim, f'{featcombs.max().item()=}>={feat_dim=}'
    assert featcombs.min() >= 0
    assert feats.dtype == torch.float
        
    bins1d = torch.linspace(low+delta, high-delta, num_bins, device=seg.device)
    mesh_y, mesh_x = [mesh.flatten() for mesh in torch.meshgrid(bins1d, bins1d, indexing='ij')] # type:ignore
    output = bins1d.new_zeros(num_seg, num_feats, num_bins**2)
    sigmaptr = bins1d.new_tensor([sigma])

    bpg = (n + tpb - 1) // tpb
    _todl = cupy.from_dlpack
    with cupy.cuda.Device(seg.device.index) as cpdev:
        _scatter_joint_hist_kernel(
            (bpg,), (tpb,), (
                _todl(seg), 
                _todl(feats),
                _todl(mesh_y),
                _todl(mesh_x),
                _todl(featcombs),
                _todl(output),
                _todl(sigmaptr),
                n, num_bins, num_feats, feat_dim
            )
        )
    return output.view(num_seg, -1)
    