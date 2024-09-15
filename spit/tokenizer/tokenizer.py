import torch
import torch.nn.functional as F

from typing import Sequence, Optional

from .proc import (
    kuma_contrast1_, asinh_contrast_, 
    apply_color_transform, scharr_features,
    _in1k_unnorm
)
from ..utils.indexing import unravel_index, fast_uidx_long2d
from ..utils.scatter import scatter_mean_2d, scatter_add_1d
from ..utils.scatterhist import scatter_hist, scatter_joint_hist
from ..utils.cossimargmax import cosine_similarity_argmax
from ..utils.concom import connected_components


def getmem(d) -> float:
    '''Returns reserved memory on device.

    Args:
        d (torch.device): A torch device.
    
    Returns:
        float: Currently reserved memory on device.
    '''
    if d.type == 'cpu':
        return 0
    a, b = torch.cuda.mem_get_info(d)
    return (b-a) / (1024**2)


def init_segedges(img:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor,int]:
    '''Computes edges and initial segmentation for an image.

    Args:
        img (torch.Tensor): Input image of shape [B,C,H,W].

    Returns:
        tuple[torch.Tensor]: tuple of segmentation, edges, and number of elements.
    '''    
    nb, _, h, w = img.shape
    nnz = nb*h*w
    seg = torch.arange(nnz, device=img.device).view(nb, h, w)
    lr = seg.unfold(-1, 2, 1).reshape(-1, 2).mT
    ud = seg.unfold(-2, 2, 1).reshape(-1, 2).mT
    edges = torch.cat([lr, ud], -1)
    return seg.view(-1), edges, nnz


def img_coords(img:torch.Tensor):
    '''Returns image coordinates.
    
    Args:
        img (torch.Tensor): Input image.

    Returns:
        torch.Tensor: Image coordinates of shape `(b,h,w)`.
    '''
    nb, _, h, w = img.shape
    return unravel_index(
        torch.arange(nb*h*w, device=img.device), # type:ignore
        (nb,h,w)
    )


def get_hierarchy_sublevel(
    hierograph:Sequence[torch.Tensor], level1:int, level2:int
) -> torch.Tensor:
    '''Retrieves superpixel mapping between level1 to level2.
    
    Args:
        hierograph (List[torch.Tensor]): List of mapping tensors.
        level1 (int): Level to compute mapping from.
        level2 (int): Level to compute mapping to.

    Returns:
        torch.Tensor: Mapping of indices from level1 to level2.
    '''
    nlvl = len(hierograph)
    if not (0 <= level1 < nlvl and 0 <= level2 < nlvl):
        raise ValueError("Invalid hierarchy levels")

    if level1 == level2:
        return hierograph[level1]
    
    min_level, max_level = min(level1, level2), max(level1, level2)

    segmentation = hierograph[min_level]
    for i in range(min_level + 1, max_level + 1):
        segmentation = hierograph[i][segmentation]

    return segmentation


def get_hierarchy_level(hierograph:Sequence[torch.Tensor], level:int) -> torch.Tensor:
    '''Retrieves superpixel mapping from initial pixels to level.
    
    Args:
        hierograph (Iterable[torch.Tensor]): List of mapping tensors.
        level (int): Level to compute mapping to.

    Returns:
        torch.Tensor: Mapping of indices up to level.
    '''
    return get_hierarchy_sublevel(hierograph, 0, level)


def preprocess_features(
    img:torch.Tensor, coords:torch.Tensor, lambda_delta:float, drop_delta:bool=False
) -> torch.Tensor:
    '''Preprocesses image for feature extraction.
    
    Args:
        img (torch.Tensor): Input image.
        lamdba_delta (float): Hyperparameter for gradient contrast adjustment.
        drop_delta (bool, optional): Flag for skipping gradient features.
        
    Returns:
        torch.Tensor: Preprocessed features.
    '''
    shape_proc = lambda x: x.permute(1,0,2,3).reshape(x.shape[1], -1).unbind(0)
    
    nb, _, h, w = img.shape
    den = max(h, w)
    _, y, x = coords.to(img.dtype).mul(2/den).sub_(1).to(img.dtype)
    r, g, b = shape_proc(img)
    kuma_contrast1_(r, .485, .539).mul_(2).sub_(1)
    kuma_contrast1_(g, .456, .507).mul_(2).sub_(1)
    kuma_contrast1_(b, .406, .404).mul_(2).sub_(1)
    features = [r,g,b,y,x]

    if not drop_delta:
        gy, gx = shape_proc(scharr_features(img, lambda_delta))
        features = [*features, gy, gx]

    return torch.stack(features, -1)


def preprocess_segmentation(seg:torch.Tensor, coords:torch.Tensor) -> torch.Tensor:
    '''Preprocesses image segmentation.
    
    Applied when segmentation indices are not unique across batches.
    
    Args:
        seg (torch.Tensor): Segmentation map of shape (B,H,W).
        coords (torch.Tensor): Tensor of shape (3,B*H*W) of pixel coords.

    Returns:
        torch.Tensor: Segmentation with unique indices across batches.
    '''
    nb, h, w = seg.shape
    shifts = torch.arange(nb, device=seg.device).mul_(h*w).view(-1, 1, 1)
    return (seg + shifts).unique(return_inverse=True)[1]


def histogram_1d(
    X:torch.Tensor, seg:torch.Tensor, num_bins:int, dims:Sequence[int],
    sigma:float, tpb:int=1024
) -> torch.Tensor:
    '''Computes 1d histogram features for selected dimensions.

    Args:
        X (torch.Tensor): Features of shape (B*H*W,D).
        seg (torch.Tensor): Segmentation map of shape (B,H,W).
        num_bins (int): Number of bins.
        dims (Iterable[int]): Feature dimensions to compute histograms for.
        sigma (float): Sigma for KDE.
        tpb (int): Threads per block.
    
    Returns:
        torch.Tensor: 1d histogram features.
    '''
    seg = seg.view(-1)
    tdims = seg.new_tensor(dims) if not torch.is_tensor(dims) else dims
    den = seg.bincount().to(dtype=X.dtype).unsqueeze(-1).mul_(len(tdims)/16)
    out = scatter_hist(seg, X[:,tdims].clone(), num_bins, sigma=sigma, tpb=tpb)
    return out / den


def histogram_2d(
    X:torch.Tensor, seg:torch.Tensor, num_bins:int, dims:Sequence[tuple[int,int]],
    sigma:float, tpb:int=1024    
) -> torch.Tensor:
    '''Computes 2d histogram features for selected dimensions.
    
    NOTE: The dimensions are pairs of dimensions we want the joint histograms over.

    Args:
        X (torch.Tensor): Features of shape (B*H*W,D).
        seg (torch.Tensor): Segmentation map of shape (B,H,W).
        num_bins (int): Number of bins.
        dims (Iterable[tuple[int,int]]): Pairs of dimensions for computing histograms.
        sigma (float): Sigma for KDE.
        tpb (int): Threads per block.
    
    Returns:
        torch.Tensor: 2d histogram features.
    '''
    seg = seg.view(-1)
    m = int(seg.max().item())+1
    den = seg.bincount().to(dtype=X.dtype).unsqueeze(-1).mul_(1/4*(num_bins/16)**2)
    out = scatter_joint_hist(seg, X, m, num_bins, dims, sigma=sigma, tpb=tpb)
    return out / den


def bbox_coords(seg:torch.Tensor, coords:torch.Tensor) -> torch.Tensor:
    '''Calculates the bounding box coordinates for each partition in segmentation maps.
    
    NOTE: Uses ordering convention ymin, xmin, ymax, xmax

    Args:
        seg (torch.Tensor): Segmentation map of shape (B, H, W).
        coords (torch.Tensor): Tensor of shape (3,B*H*W) of pixel coords.

    Returns:
        torch.Tensor: bbox coordinates (ymin, xmin, ymax, xmax) for each partition.
    '''    
    nb, h, w = seg.shape
    _, y, x = coords
    bbox = seg.new_zeros(4, int(seg.max().item()) + 1)
    bbox[0].scatter_reduce_(0, seg.view(-1), y, 'amin', include_self=False)
    bbox[1].scatter_reduce_(0, seg.view(-1), x, 'amin', include_self=False)
    bbox[2].scatter_reduce_(0, seg.view(-1), y, 'amax', include_self=False)
    bbox[3].scatter_reduce_(0, seg.view(-1), x, 'amax', include_self=False)        
    return bbox


def bbox_interpolate(
    feat:torch.Tensor, seg:torch.Tensor, coords:torch.Tensor, 
    num_bins:int, dims:Sequence[int], mode:str='bilinear'
) -> torch.Tensor:
    '''Interpolation of partition to fixed square size.

    This function assumes that the dimensions specified in dims are the last two dimensions 
    of the input tensor X. It uses bilinear or nearest neighbour interpolation and generates 
    a mask for each bounding box to ensure the interpolation only affects the pixels within 
    the bounding box.

    Args:
        X (torch.Tensor): Features of shape (B*H*W,D).
        seg (torch.Tensor): Segmentation map of shape (B,H,W).
        coords (torch.Tensor): Tensor of shape (3,B*H*W) of pixel coords.
        num_bins (int): Dimension of the square interpolation.
        dims (Iterable[int]): An iterable of dimensions / channels to interpolate.
        mode (str): The interpolation mode, either `nearest` or `bilinear`.

    Returns:
        torch.Tensor: A tensor of square bilinearly interpolated features.
    '''
    assert mode in ['bilinear', 'nearest']
    nb, h, w = seg.shape
    b, y, x = coords
    dims = seg.new_tensor(dims) if not torch.is_tensor(dims) else dims # type: ignore
    
    # Construct the batch indices of the segmentation
    b_idx = seg.view(-1).mul(nb).add(b).unique() % nb
    
    # Construct image and bbox coordinates
    img = feat[:,dims].view(nb, h, w, -1)
    ymin, xmin, ymax, xmax = bbox_coords(seg, coords).view(4, -1, 1, 1)
    
    # Construct the grid
    grid_base = torch.linspace(0, 1, num_bins, device=feat.device, dtype=feat.dtype)
    ygrid, xgrid = torch.meshgrid(grid_base, grid_base, indexing='ij')
    ygrid, xgrid = ygrid.reshape(-1, num_bins**2, 1), xgrid.reshape(-1, num_bins**2, 1)

    # Get coordinates and indices for batch / channel dimensions
    h_pos = ygrid * (ymax - ymin) + ymin
    w_pos = xgrid * (xmax - xmin) + xmin
    b_idx = b_idx.view(-1, 1, 1).expand(-1, num_bins**2, -1)
    c_idx = dims.view(1,1,-1).expand(*b_idx.shape[:2], -1) # type: ignore
    
    if mode == 'bilinear':
        
        # Construct lower and upper bounds
        h_floor = h_pos.floor().long().clamp(0, h-1)
        w_floor = w_pos.floor().long().clamp(0, w-1)
        h_ceil = (h_floor + 1).clamp(0, h-1)
        w_ceil = (w_floor + 1).clamp(0, w-1)
        
        # Construct fractional parts of bilinear coordinates
        Uh, Uw = h_pos - h_floor, w_pos - w_floor
        Lh, Lw = 1 - Uh, 1 - Uw
        hfwf, hfwc, hcwf, hcwc = Lh*Lw, Lh*Uw, Uh*Lw, Uh*Uw
        
        # Get interpolated features
        bilinear = (
            img[b_idx, h_floor, w_floor, c_idx] * hfwf +
            img[b_idx, h_floor, w_ceil, c_idx] * hfwc +
            img[b_idx, h_ceil,  w_floor, c_idx] * hcwf +
            img[b_idx, h_ceil,  w_ceil, c_idx] * hcwc
        )
        
        # Get masks
        srange = torch.arange(b_idx.shape[0], device=feat.device).view(-1,1)
        masks = (
            (seg[b_idx[:,:,0], h_floor[:,:,0], w_floor[:,:,0]] == srange).unsqueeze(-1) * hfwf +
            (seg[b_idx[:,:,0], h_floor[:,:,0], w_ceil[:,:,0]] == srange).unsqueeze(-1) * hfwc +
            (seg[b_idx[:,:,0], h_ceil[:,:,0], w_floor[:,:,0]] == srange).unsqueeze(-1) * hcwf +
            (seg[b_idx[:,:,0], h_ceil[:,:,0], w_ceil[:,:,0]] == srange).unsqueeze(-1) * hcwc
        )
        
        return (bilinear * masks).view(bilinear.shape[0], -1)
    
    elif mode == 'nearest':
        # Construct lower and upper bounds
        h_pos = h_pos.round().long().clamp(0, h-1)
        w_pos = w_pos.round().long().clamp(0, w-1)

        # Get interpolated features
        nearest = img[b_idx, h_pos, w_pos, c_idx]
        
        # Get masks
        srange = torch.arange(b_idx.shape[0], device=feat.device).view(-1,1)
        mask = (seg[b_idx[:,:,0], h_pos[:,:,0], w_pos[:,:,0]] == srange).unsqueeze(-1)

        return (nearest * mask).view(nearest.shape[0], -1)
    
    raise ValueError(f'Invalid interpolation mode: {mode=}')
    

def postprocess_for_attention(
    feat:torch.Tensor, seg:torch.Tensor, coords:torch.Tensor
) -> tuple[torch.Tensor,...]:
    '''Postprocess features for self-attention operators.

    Essentially computes an attention mask for non-fixed numbers of patches and pads
    the features to accept a global class token.

    Args:        
        feat (torch.Tensor): Features of shape (B*H*W,D).
        seg (torch.Tensor): Segmentation map of shape (B,H,W).
        coords (torch.Tensor): Tensor of shape (3,B*H*W) of pixel coords.

    Returns:
        tuple[torch.Tensor]: Output features, attention mask, global- and batch indices.
    '''
    nb, b = seg.shape[0], coords[0]
    b_idx = seg.view(-1).mul(nb).add(b).unique() % nb
    bc = b_idx.bincount()
    maxdim = bc.max() + 1
    idx = (
        torch.arange(len(b_idx), device=b_idx.device) - 
        (bc.cumsum(-1) - bc).repeat_interleave(bc)
    )

    amask = feat.new_zeros(nb, maxdim, dtype=torch.bool)
    outfeat = feat.new_zeros(nb, maxdim, feat.shape[-1])
    amask[b_idx, idx+1] = True
    amask[:,0] = True
    outfeat[b_idx, idx+1] = feat
    g_idx = torch.arange(0, nb*maxdim, maxdim, device=idx.device)
    b_idx = torch.arange(nb*maxdim, device=b_idx.device) // maxdim

    return outfeat.view(-1, feat.shape[-1]), amask, g_idx, b_idx


def init_img_graph(
    img:torch.Tensor, lambda_delta:float, lambda_col:float, drop_delta:bool=False
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], 
           torch.Tensor, torch.Tensor, int, torch.Tensor]:
    '''Initialises image graph and base level features.

    Args:
        img (torch.Tensor): Image tensor.
        grad_contrast (float): Parameter for gradient contrast.
        col_contrast (float): Parameter for color contrast.
        drop_delta (bool, optional): Drop computation of discrete gradient features. Default: False

    Returns:
        tuple[torch.Tensor]: tuple with image graph features.
    '''
    nb, _, h, w = img.shape
    lab, edges, nnz = init_segedges(img)
    coords = unravel_index(lab, (nb, h, w))
    feat = preprocess_features(img, coords, lambda_delta, False) # Keep delta for vfeat
    sizes = torch.ones_like(lab)
    maxval = asinh_contrast_(img.new_tensor(13/16), lambda_delta).mul_(2**.5)
    vfeat = torch.cat([
        apply_color_transform(feat[:,:3], img.shape, lambda_col), 
        feat[:,-2:].norm(2, dim=1, keepdim=True).div_(maxval).mul_(2).sub_(1),
    ], -1).float()

    if drop_delta:
        feat = feat[:,:5]

    return (
        lab, edges, vfeat, feat, sizes, nnz, coords
    )


def spit_step(
    lab:torch.Tensor, edges:torch.Tensor, vfeat:Optional[torch.Tensor],
    sizes:torch.Tensor, nnz:int, lvl:int, bbox:Optional[torch.Tensor]=None, tpb:int=1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor, Optional[torch.Tensor]]:
    '''Computes a Superpixel Hierarchy step.

    Args:
        lab (torch.Tensor): Labels, segmentation.
        edges (torch.Tensor): Edges.
        vfeat (torch.Tensor): Superpixel features.
        sizes (torch.Tensor): Superpixel sizes.
        nnz (int): No. vertices.
        lvl (int): Level of superpixel hierarchy.
        tpb (int, optional): CUDA threads per block. Default: 1024

    Returns:
        tuple[torch.Tensor]: Updated parameters.
    '''
    assert vfeat is not None
    sim = cosine_similarity_argmax(vfeat, edges, sizes, nnz, tpb=tpb, lvl=None, bbox=bbox)
    cc = connected_components(lab, sim, nnz, tpb=tpb)
    vfeat_new = scatter_mean_2d(vfeat, cc)
    edges_new = cc[edges].contiguous()
    edges_new = edges_new[:, fast_uidx_long2d(edges_new)]
    lab_new = cc.unique()
    nnz_new = len(lab_new)
    sizes_new = scatter_add_1d(sizes, cc, nnz_new)
    bbox_new = None
    if bbox is not None:
        bbox_new = bbox.new_zeros(4, nnz_new)
        bbox_new[0].scatter_reduce_(0, cc, bbox[0], 'amin', include_self=False)
        bbox_new[1].scatter_reduce_(0, cc, bbox[1], 'amin', include_self=False)
        bbox_new[2].scatter_reduce_(0, cc, bbox[2], 'amax', include_self=False)
        bbox_new[3].scatter_reduce_(0, cc, bbox[3], 'amax', include_self=False)
    
    return lab_new, edges_new, vfeat_new, sizes_new, nnz_new, cc, bbox_new


def _aggstep(cc, edges, vfeat, sizes, nnz, bbox=None, do_feat=True):
    vfeat_new = None
    if do_feat:        
        vfeat_new = scatter_mean_2d(vfeat, cc)
    edges_new = cc[edges]
    edges_new = edges_new[:, fast_uidx_long2d(edges_new)].contiguous()
    nnz_new = cc.max().item() + 1
    lab_new = torch.arange(nnz_new, device=cc.device)
    sizes_new = scatter_add_1d(sizes, cc, nnz_new)
    bbox_new = None
    if bbox is not None:
        bbox_new = bbox.new_zeros(4, nnz_new)
        bbox_new[0].scatter_reduce_(0, cc, bbox[0], 'amin', include_self=False)
        bbox_new[1].scatter_reduce_(0, cc, bbox[1], 'amin', include_self=False)
        bbox_new[2].scatter_reduce_(0, cc, bbox[2], 'amax', include_self=False)
        bbox_new[3].scatter_reduce_(0, cc, bbox[3], 'amax', include_self=False)
    return lab_new, edges_new, vfeat_new, sizes_new, nnz_new, cc, bbox_new


def _finalstep(th, edges, vfeat, sizes, nnz, bbox, tpb=1024):
    u, v = edges
    diff = vfeat[u] - vfeat[v]
    mask = (diff.max(-1).values - diff.min(-1).values).abs() < th
    cc = connected_components(u[mask], v[mask], nnz, tpb=tpb)
    return _aggstep(cc, edges, vfeat, sizes, nnz, bbox)


def superpixel_tokenizer(
    img:torch.Tensor, lambda_grad:float, lambda_col:float, drop_delta:bool, 
    bbox_reg:bool=False, debug:bool=False, maxlvl:int=4, tpb:int=1024,
    final_th:float=0.0, deactivate_in1k_unnorm:bool=False
):
    '''Superpixel tokenizer and feature preprocessor.

    Args:
        img (torch.Tensor): Image of shape [B, 3, H, W].
        lambda_grad (float): Lambda for gradient. 
        lambda_col (float): Lambda for color. 
        drop_delta (float): Drop discrete gradient features. 
        bbox_reg: (bool): Whether to use bounding box compactness regularization.
        debug (bool): Print debug info for superpixel iterations.
        maxlvl (int): Max number of levels. Defaults to 4.
        tbp (int): Threads per block for cuda computations.
        final_th (float): Use final thresholding.
        deactivate_in1k_unnorm (bool): Force tokenizer to ignore normalization. Defaults to False.
    '''
    if not deactivate_in1k_unnorm:
        if (img < 0).any():
            img = _in1k_unnorm(img, 1)

    # Assert 3 channels
    assert img.shape[1] == 3

    # Init variables
    device = img.device
    batch_size, _, height, width = img.shape    
    lab, edges, vfeat, feat, sizes, nnz, coords = init_img_graph(
        img, lambda_grad, lambda_col, drop_delta
    )
    bbox = None
    if bbox_reg:
        bbox = torch.stack([coords[1], coords[2], coords[1], coords[2]], 0)
    
    hierograph = [lab]
    lvl = 0
    
    # If debug, check before main loop
    if debug:
        print(f"lvl:{lvl:3} nnz:{nnz:12}  mu:{sizes.float().mean().item():8.9f} mem:{getmem(device):8}")

    # Main loop
    while lvl < maxlvl:
        
        lvl += 1
        lab, edges, vfeat, sizes, nnz, cc, bbox = spit_step(
            lab, edges, vfeat, sizes, nnz, lvl, bbox, tpb
        )
        hierograph.append(cc)
        if debug:
            print(f"lvl:{lvl:3} nnz:{nnz:12}  mu:{sizes.float().mean().item():8.9f} mem:{getmem(device):8}")

    if final_th > 0:

        lvl += 1
        lab, edges, vfeat, sizes, nnz, cc, bbox = _finalstep(
            final_th, edges, vfeat, sizes, nnz, bbox, tpb
        )
        hierograph.append(cc)

    # Compile segmentation from hierarchy
    seg = get_hierarchy_level(hierograph, lvl)
    return feat, seg.view(batch_size, height, width), coords


def random_rectangular_partitions(
    b:int, h:int, w:int, p_low:int, p_high:int, 
    roll:bool=True, square:bool=True,
    device:torch.device=torch.device('cpu')
) -> torch.Tensor:
    '''Generates random square partitions of dimension BxHxW.
    
    Args:
        b (int): Batch size.
        h (int): Raster height.
        w (int): Raster width.
        p_low (int): Minimum partition size.
        p_high (int): Maximum partition size (inclusive).
        roll (bool): Flag for randomized rolling of rows and columns.
        square (bool): Flag for enforcing square partitions.
        device (torch.device): Output device.
    
    Returns:
        torch.Tensor: Square partition.
        
    '''
    def _quickroll(A:torch.Tensor, shifts:tuple[torch.Tensor,torch.Tensor]):
        shape = A.shape
        for i in range(2):
            A = A.mT.reshape(-1, shape[-2 + i])
            rng = torch.arange(shape[-2 + i], device=A.device).view(1,-1).expand_as(A)
            idx = (rng + shifts[i].view(-1,1)) % shape[-2 + i]
            A = A.gather(1, idx).view(*shape)
        return A
    
    ps_h, ps_w = torch.randint(p_low, p_high+1, (2,b), device=device)
    ps_w = ps_h if square else ps_w
    ceil_h, ceil_w = -(-h//ps_h), -(-w//ps_w)
    ceil_hw = ceil_h * ceil_w

    partition_ids = torch.arange(ceil_hw.sum().item(), device=device)
    bs = torch.arange(b, device=device).repeat_interleave(ceil_hw)
    idx = partition_ids - (ceil_hw.cumsum(-1) - ceil_hw)[bs]
    y, x = idx // ceil_w[bs], idx % ceil_w[bs]
    batched_x = x + (ceil_w.cumsum(-1) - ceil_w)[bs]
    
    psize_h = (
        torch.stack([ps_h, h - (ceil_h - 1)*ps_h], -1)
            .view(-1)
            .repeat_interleave(
                torch.stack([ceil_h-1, ceil_h.new_ones(b)], -1)
                    .view(-1)
            )
    )
    psize_w = (
        torch.stack([ps_w, w - (ceil_w - 1)*ps_w], -1)
            .view(-1)
            .repeat_interleave(
                torch.stack([ceil_w-1, ceil_w.new_ones(b)], -1)
                    .view(-1)
            )
    )
        
    out = (
        partition_ids
            .repeat_interleave(psize_w[batched_x])
            .view(-1, w)
            .repeat_interleave(psize_h, dim=0)
            .view(b, h, w)
    )

    if not roll:
        return out

    roll_h = (
        torch.rand_like(ceil_h.float())
            .mul_(ceil_h)
            .long()
            .mul_(ps_h)
            .repeat_interleave(h)
    )
    roll_w = (
        torch.rand_like(ceil_w.float())
            .mul_(ceil_w)
            .long()
            .mul_(ps_w)
            .repeat_interleave(w)
    )
    return _quickroll(out, (roll_h, roll_w))
