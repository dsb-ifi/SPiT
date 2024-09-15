import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from torch.distributions import Beta
from typing import Optional, Tuple

from .tokenizer.tokenizer import (
    superpixel_tokenizer, preprocess_features, preprocess_segmentation,
    img_coords, random_rectangular_partitions, bbox_interpolate, histogram_2d,
    postprocess_for_attention
)
from .tokenizer.voronoi import fast_pseudo_voronoi, chunked_voronoi

class SuperpixelTokenizer(nn.Module):

    def __init__(self, drop_delta:bool=False, bbox_reg=False, **kwargs):
        super().__init__()
        self.drop_delta = drop_delta
        self._lgrad = 27.8
        self._lcol = 10.
        self._maxlvl = 4
        self._bbox = bbox_reg
        self._final_th = 0.0

    def forward(self, feat:torch.Tensor, *args):
        return superpixel_tokenizer(
            feat, self._lgrad, self._lcol, self.drop_delta, 
            bbox_reg=self._bbox, maxlvl=self._maxlvl,
            final_th=self._final_th
        )
    
    def enable_bbox(self, lgrad:float=27.8, lcol:float=10.0):
        self._bbox = True
        self._lgrad = lgrad
        self._lcol = lcol

    def set_final_th(self, th):
        self._final_th = th


class DefaultTokenizer(nn.Module):

    def __init__(
        self, num_bins:int, p_low:int=12, p_high:int=32, roll:bool=False, 
        drop_delta:bool=False, mode:str='bilinear', rvt:bool=False, prvt:bool=False, **kwargs
    ):
        super().__init__()            
        self.p_low = p_low
        self.p_high = p_high
        self.roll = roll
        self.drop_delta = drop_delta
        self.prvt = prvt
        self.rvt = rvt
        self.num_bins = num_bins
        if mode == 'nearest':
            self.p_low = num_bins
            self.p_high = num_bins
            self.roll = False 

    def forward(self, img:torch.Tensor, seg:Optional[torch.Tensor]):
        nb, _, h, w = img.shape
        coords = img_coords(img)
        feat = preprocess_features(img, coords, 27.8, self.drop_delta)
        if seg is None:
            if self.rvt: 
                num_cells = int(round((h*w)**.5))
                seg = chunked_voronoi(img, num_cells)

            elif self.prvt: # Use pseudo random voronoi tesselation
                num_cells = int(round((h*w)**.5 * 1.1))
                seg = fast_pseudo_voronoi(img, num_cells)
                
            else:
                seg = random_rectangular_partitions(
                    nb, h, w, self.p_low, self.p_high, self.roll, device=feat.device
                )
        else:
            seg = preprocess_segmentation(seg, coords)
        
        return feat, seg, coords
    
    def enable_bbox(self, *args):
        raise AttributeError('Cannot use compactness regularization for ViT / RViT!')


class InterpolationExtractor(nn.Module):

    def __init__(
        self, num_bins:int, in_channels:int=3, sigma2d:float=0.025, 
        drop_delta:bool=True, tpb:int=1024, mode:str='bilinear', **kwargs
    ):
        super().__init__()
        assert mode in ['nearest', 'bilinear']
        self.num_bins = num_bins
        self.in_channels = in_channels
        self._intp_dims = tuple(range(in_channels))
        self.sigma2d = sigma2d
        self.tpb = tpb
        self.mode = mode
        if drop_delta:
            self._2d_dims = ((in_channels, in_channels+1),)
        else:
            self._2d_dims = ((in_channels, in_channels+1), (in_channels+2, in_channels+3))
        self._half2d = self.num_bins**2
        self.drop_delta = drop_delta

    def forward(self, feat:torch.Tensor, seg:torch.Tensor, coords:torch.Tensor, *args):
        tokens_intp = bbox_interpolate(
            feat, seg, coords, self.num_bins, self._intp_dims, self.mode
        ).view(-1, self._half2d, self.in_channels)

        tokens_2d = histogram_2d(
            feat, seg, self.num_bins, self._2d_dims, self.sigma2d, self.tpb
        ).view(-1, len(self._2d_dims), self._half2d).mT

        return torch.cat((tokens_intp, tokens_2d), -1).view(tokens_intp.shape[0], -1)

    def _forward_old(self, feat:torch.Tensor, seg:torch.Tensor, coords:torch.Tensor, *args):
        tokens_intp = bbox_interpolate(
            feat, seg, coords, self.num_bins, self._intp_dims, self.mode
        )

        tokens_2d = histogram_2d(
            feat, seg, self.num_bins, self._2d_dims, self.sigma2d, self.tpb
        )

        # Balance normalization of gradients for 2d
        if not self.drop_delta:
            tokens_2d.mul_(4)[:,self._half2d:].mul_(0.2)

        return torch.cat((tokens_intp, tokens_2d), -1)
    
    def interpolate_patch_size(self, new_bins):
        self.num_bins = new_bins
        self._half2d = self.num_bins**2
        self.mode = 'bilinear'


class MaskedLinear(nn.Module):

    def __init__(self, in_feat, out_feat, bias=True, activation=None):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.linear = nn.Linear(in_feat, out_feat, bias=bias)
        if activation is None:
            self.act = nn.Identity()
        else:
            self.act = activation
        if bias:
            self.linear.bias.data.mul_(1e-3)
        
    def forward(self, x, amask):
        assert x.ndim == 2
        masked_output = self.act(self.linear(x[amask.view(-1)]))
        out = torch.zeros(x.shape[0], self.out_feat, dtype=masked_output.dtype, device=x.device)
        out[amask.view(-1)] = masked_output
        return out


class TokenEmbedder(nn.Module):

    def __init__(
        self, num_bins:int, emb_dim:int, keep_k:int, extractor:str,
        in_channels:int=3, drop_delta:bool=False, **kwargs
    ):
        super().__init__()
        self.num_bins = num_bins
        self.emb_dim = emb_dim
        self.in_channels = in_channels
        self.drop_delta = drop_delta
        self.keep_k = keep_k
        self._extractorstr = extractor

        if extractor == 'histogram':
            self.token_dim = (
                in_channels * (num_bins**2 // in_channels) + num_bins**2 * (2 - drop_delta)
            )
        elif extractor == 'interpolate':
            self.token_dim = num_bins**2 * (in_channels + 2 - drop_delta)
        else:
            raise ValueError(f'Invalid extractor: {extractor:=}')
        
        self.embedder = MaskedLinear(self.token_dim, self.emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(self.emb_dim) + 1e-5)

    def budget_dropout(self, amask:torch.Tensor) -> torch.Tensor:
        '''Randomly drops features keeping max token size of k, leaving the first elements untouched.
        
        Args:
            amask (torch.Tensor): Attention mask

        Returns:
            torch.Tensor: Indices of tokens to keep after budget dropout.
        '''
        b, t = amask.shape
        first = torch.arange(b, device=amask.device) * t
        keep = torch.multinomial(amask[:, 1:].float(), t - 1)  + 1
        b_idx = torch.arange(b, device=keep.device).view(-1, 1).expand_as(keep) * t
        keep = torch.cat((first.view(-1,1), (b_idx + keep)[:, :self.keep_k - 1]), dim=1)    
        return keep.reshape(-1).sort().values
    
    def forward(self, feat:torch.Tensor, seg:torch.Tensor, coords:torch.Tensor, *args):
        nb = seg.shape[0]
        feat, amask, g_idx, b_idx = postprocess_for_attention(feat, seg, coords)

        if self.keep_k > 0 and self.training:
            keep = self.budget_dropout(amask)
            feat = feat[keep]
            amask = amask[:,:self.keep_k].contiguous()
            b_idx = b_idx[keep].contiguous()
            g_idx = torch.arange(0, nb*self.keep_k, self.keep_k, device=g_idx.device)

        feat = self.embedder(feat, amask)
        feat[g_idx] = self.cls_token.view(1, -1).expand(nb, -1)
        return feat, amask, g_idx, b_idx
    
    def random_sample_embed(self, feat, amask, g_idx, b_idx, keep_k, drop=True):
        nb = amask.shape[0]
        keep = None
        if drop:
            # Replace keep_k
            old_keep = self.keep_k
            self.keep_k = keep_k

            # Drop non_keeps
            keep = self.budget_dropout(amask)
            feat = feat[keep].clone()
            amask = amask[:,:self.keep_k].clone()
            b_idx = b_idx[keep].clone()
            g_idx = torch.arange(0, nb*self.keep_k, self.keep_k, device=g_idx.device)

            # Reset original keep_k
            self.keep_k = old_keep

        # Compute current outputs
        feat = self.embedder(feat, amask)
        feat[g_idx] = self.cls_token.view(1, -1).expand(nb, -1)
        return feat, amask, g_idx, b_idx, keep


class MaskedMSA(nn.Module):
        
    def __init__(
        self, embed_dim:int, heads:int, dop_att:float=0.0, dop_proj:float=0.0,
        qkv_bias:bool=False, lnqk:bool=False, **kwargs
    ):
        super().__init__()
        assert embed_dim % heads == 0, f'Invalid args: embed_dim % heads != 0.'
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.scale = self.head_dim ** -.5
        self.dop_att = dop_att
        self.dop_proj = dop_proj
        self.qkv = MaskedLinear(embed_dim, 3*embed_dim, bias=qkv_bias)
        self.proj = MaskedLinear(embed_dim, embed_dim)
        if lnqk:
            self.ln_k = nn.LayerNorm(self.head_dim, eps=1e-6)
            self.ln_q = nn.LayerNorm(self.head_dim, eps=1e-6)
        else:
            self.ln_k = nn.Identity()
            self.ln_q = nn.Identity()
        
    def doo(self, x):
        return F.dropout(x, self.dop_proj, training=self.training)

    def doa(self, x):
        return F.dropout(x, self.dop_att, training=self.training)
    
    def expand_mask(self, amask):
        m, n = amask.shape
        return amask.view(m,1,1,n).expand(m, self.heads, n, n)
        
    def forward(self, feats, amask, store_att=False, pre_softmax=False):
        b, t = amask.shape
        h, d = self.heads, self.head_dim
        n, c = feats.shape
        m = n - b*t

        out = torch.zeros_like(feats)
        q, k, v = (
            self.qkv(feats[m:], amask)
                .view(b, t, 3, h, d)
                .permute(2,0,3,1,4)
        )
        if not store_att:
            out[m:] = self.proj(
                F.scaled_dot_product_attention(
                    self.ln_q(q), self.ln_k(k), v
                ).transpose(1,2).reshape(-1, c),
                amask
            )
            return self.doo(out)
        
        else:
            out[m:] = self.proj(
                self._manual_att(
                    self.ln_q(q), self.ln_k(k), v, amask, pre_softmax
                ).transpose(1,2).reshape(-1, c),
                amask
            )
            return self.doo(out)
        

class MaskedMLP(nn.Module):

    def __init__(self, embed_dim:int, hid_dim:int, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.L1 = MaskedLinear(embed_dim, hid_dim, activation=nn.GELU())
        self.L2 = MaskedLinear(hid_dim, embed_dim)
    
    def forward(self, x:torch.Tensor, amask:torch.Tensor):
        x = self.L1(x, amask)
        return self.L2(x, amask)


class LayerScale(nn.Module):

    def __init__(self, embed_dim:int, init_val:float=1e-5):
        super().__init__()
        self.lambda_ = nn.Parameter(torch.full((embed_dim,), init_val))

    def forward(self, x):
        return x * self.lambda_
    

class DropPath(nn.Module):

    def __init__(self, p:float, scale_by_keep:bool=True):
        super().__init__()
        self.p = p
        self.q = 1 - p
        self.scale_by_keep = scale_by_keep
        
        
    def forward(self, x:torch.Tensor, batch_idx:Optional[torch.Tensor]=None):
        if self.p == 0 or not self.training:
            return x
        
        if batch_idx is None:
            shape = (x.size(0), *((1,)*(x.ndim-1)))
            drops = x.new_empty(*shape).bernoulli_(self.q)

            if self.q > 0. and self.scale_by_keep:
                drops.div_(self.q)

            return x * drops
        
        nb = (batch_idx.max() + 1).item()
        drops = x.new_empty(nb).bernoulli_(self.q) # type: ignore

        if self.q > 0. and self.scale_by_keep:
            drops.div_(self.q)

        return x * drops.gather(0, batch_idx)[:,None]


class MaskedViTBlock(nn.Module):

    def __init__(
        self, embed_dim, heads, mlp_ratio=4.0, dop_path:float=0.0, use_cp=False, **kwargs
    ):
        super().__init__()
        self.use_cp = use_cp
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ls1 = LayerScale(embed_dim)
        self.ls2 = LayerScale(embed_dim)
        self.dop1 = DropPath(dop_path)
        self.dop2 = DropPath(dop_path)
        hid_dim = int(embed_dim * mlp_ratio)
        self.att = MaskedMSA(embed_dim, heads, **kwargs)
        self.mlp = MaskedMLP(embed_dim, hid_dim)

    def _fwd(self, x, amask, batch_idx, store_att=False, pre_softmax=False):
        x = x + self.dop1(self.ls1(self.att(self.norm1(x), amask, store_att=store_att, pre_softmax=pre_softmax)), batch_idx)
        x = x + self.dop2(self.ls2(self.mlp(self.norm2(x), amask)), batch_idx)
        return x    

    def forward(self, x, amask, batch_idx, store_att=False, pre_softmax=False):
        if self.use_cp and self.training:
            return checkpoint( 
                self._fwd, x, amask, batch_idx, store_att=store_att, pre_softmax=pre_softmax
            )
        return self._fwd(x, amask, batch_idx, store_att=store_att, pre_softmax=pre_softmax)


class SPiT(nn.Module):

    def __init__(
        self, emb_dim:int, num_bins:int, heads:int, depth:int, classes:int, keep_k:int, 
        extractor:str, tokenizer:str, in_channels:int=3, dop_input:float=0.0, **kwargs
    ):
        super().__init__()

        # Initialize tokenizer
        if tokenizer == 'default':
            self.tokenizer = DefaultTokenizer(num_bins, **kwargs)
        elif tokenizer == 'superpixel':
            self.tokenizer = SuperpixelTokenizer(**kwargs)
        else:
            raise ValueError(f'Invalid argument: {tokenizer=}')

        # Initialize extractor
        if extractor == 'interpolate':
            self.extractor = InterpolationExtractor(num_bins, in_channels, **kwargs)
        else:
            raise ValueError(f'Invalid argument: {extractor=}')
        
        self.embedder = TokenEmbedder(num_bins, emb_dim, keep_k, extractor, in_channels, **kwargs)
        self.dop_input = dop_input
        self.blocks = nn.ModuleList([
            MaskedViTBlock(
                emb_dim,
                heads,
                **kwargs
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.head = nn.Linear(emb_dim, classes)
        self.id = nn.Identity()
        self.segmodel = False
        if 'segmodel' in kwargs:
            self.segmodel =  kwargs['segmodel']

    def doi(self, x):
        return F.dropout(x, self.dop_input, training=self.training)
    
    def avg_noncls_tokens(self, feat:torch.Tensor, amask:torch.Tensor, g_idx:torch.Tensor):
        b_idx = torch.where(amask)[0].unsqueeze(1).expand(-1, feat.shape[1])
        out = -feat[g_idx]
        out.scatter_add_(
            0, b_idx, feat[amask.view(-1)]
        ).div_(amask.sum(1).view(-1, 1) - 1)
        return out
    
    def to_segmodel(self, seg_classes:int):
        self.to_newclasses(seg_classes)
        self.segmodel = True

    def to_newclasses(self, new_classes:int):
        emb_dim = self.head.in_features
        device = self.head.weight.device
        self.head = nn.Linear(emb_dim, new_classes, device=device)

    @staticmethod
    def maxmin(x, eps=1e-6):
        mx, mn = x[:,1:].max(-1, keepdim=True).values, x[:,1:].min(-1, keepdim=True).values
        return (x - mn) / (mx - mn + eps)
        
    def normalized_attmap(self, attn):
        attmap = attn[:,0].log()
        std_dev = torch.std(attmap, dim=-1, keepdim=True)
        mean = torch.mean(attmap, dim=-1, keepdim=True)
        attmap = (attmap - mean) / std_dev
        return attmap
        
    def normalized_pca(self, feat, prototypes, num_pc=1):
        if prototypes is not None:
            feat = torch.cat([prototypes, feat], 1)
        mean = torch.mean(feat, dim=1, keepdim=True)
        feat_centered = feat - mean
        if prototypes is not None:
            feat_centered = feat_centered[:,prototypes.shape[1]:]
        _, _, V = torch.pca_lowrank(feat_centered, num_pc)
        proj = torch.einsum('bnd,bdp->bnp', feat_centered, V).max(-1).values
        std_dev = torch.std(proj[:,1:], dim=-1, keepdim=True)
        mean = torch.mean(proj[:,1:], dim=-1, keepdim=True)
        proj[:,1:] = (proj[:,1:] - mean) / std_dev
        return proj

    def forward(
        self, feat:torch.Tensor, seg:Optional[torch.Tensor] = None, 
        headless=False, return_seg=None, return_attn=False, return_pca=False, 
        prototypes=None, do_lazy=False, newtok=False, parttok=False, 
        save_feats=None, **kwargs
    ):  
        if do_lazy:
            return torch.inverse(torch.ones((0, 0), device="cuda:0"))
        if not newtok:
            if parttok:
                feat, seg, coords, sizes, nnz, bbox = self.tokenizer(feat)
                feat, seg, coords, nnz = self.extractor(feat, seg, coords, sizes, nnz, bbox)
                feat = feat.flatten(1,-1)
            else:
                feat, seg, coords = self.tokenizer(feat, seg)
                feat = self.extractor(feat, seg, coords)
            feat, amask, g_idx, b_idx = self.embedder(feat, seg, coords)
        else:
            feat, seg, coords, sizes, nnz, bbox = self.tokenizer(feat)
            feat, seg, coords, nnz = self.extractor(feat, seg, coords, sizes, nnz, bbox)
            feat, seg, amask = self.embedder(feat, seg, coords, nnz)
            amcp = amask.clone()
            amcp[:,1:] = 0
            g_idx = amcp.view(-1)
            b_idx = None
        attn = None
        pca = None
        out = []
        eye = torch.eye(amask.shape[-1], device=feat.device)
        head_fn = self.id if headless else self.head

        if return_seg is None:
            return_seg = self.segmodel

        feat = self.doi(feat)
        store_feat = []
        save_feats = [len(self.blocks)-1] if save_feats is None else save_feats
        psm = kwargs.get('pre_softmax', False)
        for _i, block in enumerate(self.blocks):
            feat = block(feat, amask, b_idx, store_att=return_attn, pre_softmax=psm)
            if _i in save_feats:
                store_feat.append(feat)
            if return_attn:
                if attn is None:
                    attn = 0.9*block.att._attn + 0.1*eye
                else:
                    attn = attn.clip(1e-8, 1) @ (0.9*block.att._attn + 0.1*eye) # Preclip 0.1
                block.att._attn = None
        feat = torch.cat([self.norm(f.view(-1, f.shape[-1])) for f in store_feat], -1)

        if not return_seg:
            out.append(head_fn(feat[g_idx]))

        else:
            assert seg is not None
            seg = seg - seg.view(seg.shape[0], -1).min(-1).values[:,None,None] + 1
            out += [head_fn(feat[amask.view(-1)]), seg]

        if attn is not None:
            if kwargs.get('attn_as_matrix', False):
                out.append(attn)
            else:
                attn = self.normalized_attmap(attn.max(1).values)
                out.append(self.maxmin(attn).view(-1)[amask.view(-1)])

        if return_pca:
            pca = self.normalized_pca(feat.view(*amask.shape, -1), prototypes)
            out.append(self.maxmin(pca).view(-1)[amask.view(-1)])

        if len(out) > 1:
            return tuple(out)
        
        return out[0]
        

    def explain(self, img, seg, label, explanations=512, keep_k_bounds=(0.1, 0.3)):
        assert len(label) == len(img)
        assert explanations > 0
        assert not self.training

        with torch.no_grad():
            feat, seg, coords = self.tokenizer(img, seg)
            feat = self.extractor(feat, seg, coords)  
            ofeat, oamask, og_idx, ob_idx = postprocess_for_attention(feat, seg, coords)
            scores = feat.new_zeros(explanations, oamask.shape[0])
            covariates = feat.new_zeros(explanations, ofeat.shape[0])
            attn = None
            rng = torch.arange(len(label), device=label.device)
            _sc, _sh = max(keep_k_bounds) - min(keep_k_bounds), min(keep_k_bounds)
            for _ex in range(explanations):
                keep_k_ratio = torch.rand(1).item() * _sc + _sh
                keep_k = int(keep_k_ratio * oamask.shape[-1])
                feat, amask, g_idx, b_idx, keep = self.embedder.random_sample_embed(ofeat, oamask, og_idx, ob_idx, keep_k)
                covariates[_ex,keep] = 1
                for block in self.blocks:
                    feat = block(feat, amask, b_idx)
                scores[_ex] = self.head(self.norm(feat)[g_idx]).softmax(-1)[rng, label]
                
            feat, amask, g_idx, b_idx, keep = self.embedder.random_sample_embed(ofeat, oamask, og_idx, ob_idx, 0, drop=False)                
            eye = torch.eye(amask.shape[-1], device=feat.device)
            for block in self.blocks:
                assert isinstance(block, MaskedViTBlock)
                ofeat = block(feat, amask, b_idx, store_att=True)
                if attn is None:
                    attn = block.att._attn.max(1).values + eye
                else:
                    attn = attn @ (block.att._attn.max(1).values + eye)
                block.att._attn = None # type: ignore
                
                    
        covariates = covariates.view(explanations, amask.shape[0], -1).permute(1,0,2)

        return self.norm(feat), attn, covariates, scores.mT
