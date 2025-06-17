import torch
import warnings
from typing import Union
from collections import OrderedDict
from .spit.nn import SPiT

dependencies = ['torch >= 2.1.0', 'torchvision >= 0.16.0', 'scipy >= 1.10.1', 'cupy >= 13.0.0', 'numba >= 0.59.0']

_architecture_cfg = {
    'S': OrderedDict(depth=12, emb_dim= 384, heads= 6, dop_path=0),
    'B': OrderedDict(depth=12, emb_dim= 768, heads=12, dop_path=0.2),
    'L': OrderedDict(depth=24, emb_dim=1024, heads=16, dop_path=0.2),
}

_std_cfg:dict[str, Union[str, int, float, bool]] = dict(
    classes = 1000, keep_k = 256, extractor = 'interpolate'
)

_modelweights_url = dict(
    SPiT_S16 = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EZ57Sad2uf9Dizwm3VYhvw4BVdHOxsEJcgyf4vgKsdmgZg',
    SPiT_S16_grad = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/Eb9FViSwap5JqYe1mtlC3jQBE-nAMG88MfJfmypT_J8r0Q',
    SPiT_B16 = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EXhsshO-DvlIii87kyyEVtoBRFbZaTp8SqTgDJhQ1iQIBw',
    SPiT_B16_grad = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EcahlrAzXZ5Bsozrqs4dWLABHFX-V5VH8jQR5ygHhZH30A',
    ViT_S16 = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EWqHDQvY5V5PjKkMmO5fcFEBKuN6WTfr4a99u8vpNT67WQ',
    ViT_S16_grad = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EenEECYQaQZFl_GeU2N9q7YB-XOHNyaJXHnC74qREU3cSQ',
    ViT_B16 = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EUWJM_RY9IRPvM9dsp2Zzi8B6ZOnhQ_C666TMESzmAQ0sQ',
    ViT_B16_grad = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EdGx5GaXRshPpOh0gsCHU4cBeZ0FxexzuBm7vTtm67nuTw',
    RViT_B16 = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/Ed9R0bQOmslLiPnFX_P0hRoBUf_zQ4pfHXZ3BpQ4iW8JYA',
    RViT_B16_grad = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EflpV7TP04RKmxg1qfiNovUBo149q0P9j4tmoOTQ-NkV-Q',
)

def _get_pretrained_weights(model:str, grad:bool=True, **kwargs):
    model_full = f'{model}_grad' if grad else model
    return torch.hub.load_state_dict_from_url(
        _modelweights_url[model_full],
        map_location="cpu",
        weights_only=True,
        **kwargs.get('torch_hub_kwargs', {})
    )

def spit_small_16(grad:bool=True, pretrained=False, **kwargs) -> SPiT:
    kwargs = {**_architecture_cfg['S'], **_std_cfg}
    kwargs['num_bins'] = 16
    kwargs['tokenizer'] = 'superpixel'
    kwargs['drop_delta'] = not grad
    kwargs['sigma2d'] = 0.025
    kwargs['bbox_reg'] = False
    if not grad:
        kwargs['bbox_reg'] = True

    model = SPiT(**kwargs)

    if pretrained:
        warnings.warn('Note that S16 weights are not fine tuned.')
        sd = _get_pretrained_weights('SPiT_S16', grad, **kwargs)
        model.load_state_dict(sd, strict=False)
        return model.eval()
    
    return model.eval()

def spit_base_16(grad:bool=True, pretrained=False, **kwargs) -> SPiT:
    kwargs = {**_architecture_cfg['B'], **_std_cfg}
    kwargs['num_bins'] = 16
    kwargs['tokenizer'] = 'superpixel'
    kwargs['drop_delta'] = not grad
    kwargs['sigma2d'] = 0.025
    kwargs['bbox_reg'] = False

    if grad:
        kwargs['sigma2d'] = 0.05
    else:
        kwargs['bbox_reg'] = True

    model = SPiT(**kwargs) # type: ignore

    if pretrained:
        sd = _get_pretrained_weights('SPiT_B16', grad, **kwargs)
        model.load_state_dict(sd)
        return model.eval()

    return model.eval()

def vit_small_16(grad:bool=True, pretrained=False, **kwargs) -> SPiT:
    kwargs = {**_architecture_cfg['S'], **_std_cfg}
    kwargs['num_bins'] = 16
    kwargs['tokenizer'] = 'default'
    kwargs['mode'] = 'nearest'
    kwargs['drop_delta'] = not grad
    kwargs['sigma2d'] = 0.025

    model = SPiT(**kwargs) # type: ignore

    if pretrained:
        warnings.warn('Note that S16 weights are not fine tuned.')
        sd = _get_pretrained_weights('ViT_S16', grad, **kwargs)
        model.load_state_dict(sd)
        return model.eval()

    return model.eval()

def vit_base_16(grad:bool=True, pretrained=False, **kwargs) -> SPiT:
    kwargs = {**_architecture_cfg['B'], **_std_cfg}
    kwargs['num_bins'] = 16
    kwargs['tokenizer'] = 'default'
    kwargs['mode'] = 'nearest'
    kwargs['drop_delta'] = not grad
    kwargs['sigma2d'] = 0.025

    model = SPiT(**kwargs) # type: ignore

    if pretrained:
        sd = _get_pretrained_weights('ViT_B16', grad, **kwargs)
        model.load_state_dict(sd)
        return model.eval()

    return model.eval()

def rvit_small_16(grad:bool=True, pretrained=False, **kwargs) -> SPiT:
    kwargs = {**_architecture_cfg['S'], **_std_cfg}
    kwargs['num_bins'] = 16
    kwargs['tokenizer'] = 'default'
    kwargs['mode'] = 'bilinear'
    kwargs['prvt'] = True
    kwargs['drop_delta'] = not grad
    kwargs['sigma2d'] = 0.025

    model = SPiT(**kwargs) # type: ignore

    if pretrained:
        raise ValueError('RViT_S16 does not have pretrained weights.')

    return model.eval()

def rvit_base_16(grad:bool=True, pretrained=False, **kwargs) -> SPiT:
    kwargs = {**_architecture_cfg['B'], **_std_cfg}
    kwargs['num_bins'] = 16
    kwargs['tokenizer'] = 'default'
    kwargs['mode'] = 'bilinear'
    kwargs['prvt'] = True
    kwargs['drop_delta'] = not grad
    kwargs['sigma2d'] = 0.025

    model = SPiT(**kwargs) # type: ignore

    if pretrained:
        sd = _get_pretrained_weights('RViT_B16', grad, **kwargs)
        model.load_state_dict(sd)
        return model.eval()

    return model.eval()

__all__ = [
    'spit_small_16',
    'spit_base_16',
    'vit_small_16',
    'vit_base_16',
    'rvit_small_16',
    'rvit_base_16',
]

