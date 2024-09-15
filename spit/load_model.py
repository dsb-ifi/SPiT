import urllib.request
import torch
import warnings
import requests
import os

from typing import Union
from collections import OrderedDict
from .nn import SPiT


_architecture_cfg = {
    'T': OrderedDict(depth=12, emb_dim= 192, heads= 3, dop_path=0),
    'S': OrderedDict(depth=12, emb_dim= 384, heads= 6, dop_path=0),
    'M': OrderedDict(depth=12, emb_dim= 512, heads= 8, dop_path=0.1),
    'B': OrderedDict(depth=12, emb_dim= 768, heads=12, dop_path=0.2),
    'L': OrderedDict(depth=24, emb_dim=1024, heads=16, dop_path=0.2),
    'H': OrderedDict(depth=32, emb_dim=1280, heads=16, dop_path=0.2),
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
    RViT_S16 = '',
    RViT_S16_grad = '',
    RViT_B16 = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/Ed9R0bQOmslLiPnFX_P0hRoBUf_zQ4pfHXZ3BpQ4iW8JYA',
    RViT_B16_grad = 'https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EflpV7TP04RKmxg1qfiNovUBo149q0P9j4tmoOTQ-NkV-Q',
)

def _download_model_weights(model: str, grad: bool = False):
    _prefix = 'SPiT_model_'
    _suffix = '_grad' if grad else ''
    model_full = f'{model}{_suffix}'
    if model_full not in _modelweights_url:
        raise KeyError(f'Invalid model: {model_full}')
    hub_dir = torch.hub.get_dir()
    local_path = os.path.join(hub_dir, 'checkpoints', f'{_prefix}{model}{_suffix}.pth')
    url = _modelweights_url[model_full] 
    if url == '':
        raise NotImplementedError('Sorry! Weights for RViT-S models have not been uploaded yet!')
    url += '?download=1'
    print(f'Downloading pretrained weights for {model_full}...')
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192): 
                f.write(chunk)
        print(f'Weights downloaded to: {local_path}')
    else:
        raise ConnectionError(
            f'Failed to download weights: HTTP status code {response.status_code}'
        )

def _get_pretrained_weights(model: str, grad: bool = False):
    _prefix = 'SPiT_model_'
    _suffix = '_grad' if grad else ''
    hub_dir = torch.hub.get_dir()
    local_path = f'{hub_dir}/checkpoints/{_prefix}{model}{_suffix}.pth'
    if not os.path.isfile(local_path):
        _download_model_weights(model, grad)
    sd = torch.load(local_path, map_location='cpu')
    return sd
    

def load_SPiT_S16(grad:bool=False, pretrained=False) -> SPiT:
    kwargs = {**_architecture_cfg['S'], **_std_cfg}
    kwargs['num_bins'] = 16
    kwargs['tokenizer'] = 'superpixel'
    kwargs['drop_delta'] = not grad
    kwargs['sigma2d'] = 0.025
    kwargs['bbox_reg'] = False
    if not grad:
        kwargs['bbox_reg'] = True

    model = SPiT(**kwargs) # type: ignore

    if pretrained:
        warnings.warn('Note that S16 weights are not fine tuned.')
        sd = _get_pretrained_weights('SPiT_S16', grad)
        model.load_state_dict(sd)
        return model.eval()
    
    return model.eval()


def load_SPiT_B16(grad:bool=False, pretrained=False) -> SPiT:
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
        sd = _get_pretrained_weights('SPiT_B16', grad)
        model.load_state_dict(sd)
        return model.eval()

    return model.eval()

def load_ViT_S16(grad:bool=False, pretrained=False) -> SPiT:
    kwargs = {**_architecture_cfg['S'], **_std_cfg}
    kwargs['num_bins'] = 16
    kwargs['tokenizer'] = 'default'
    kwargs['mode'] = 'nearest'
    kwargs['drop_delta'] = not grad
    kwargs['sigma2d'] = 0.025

    model = SPiT(**kwargs) # type: ignore

    if pretrained:
        warnings.warn('Note that S16 weights are not fine tuned.')
        sd = _get_pretrained_weights('ViT_S16', grad)
        model.load_state_dict(sd)
        return model.eval()

    return model.eval()

def load_ViT_B16(grad:bool=False, pretrained=False) -> SPiT:
    kwargs = {**_architecture_cfg['B'], **_std_cfg}
    kwargs['num_bins'] = 16
    kwargs['tokenizer'] = 'default'
    kwargs['mode'] = 'nearest'
    kwargs['drop_delta'] = not grad
    kwargs['sigma2d'] = 0.025

    model = SPiT(**kwargs) # type: ignore

    if pretrained:
        sd = _get_pretrained_weights('ViT_B16', grad)
        model.load_state_dict(sd)
        return model.eval()

    return model.eval()

def load_RViT_S16(grad:bool=False, pretrained=False) -> SPiT:
    kwargs = {**_architecture_cfg['S'], **_std_cfg}
    kwargs['num_bins'] = 16
    kwargs['tokenizer'] = 'default'
    kwargs['mode'] = 'bilinear'
    kwargs['rvt'] = True
    kwargs['drop_delta'] = not grad
    kwargs['sigma2d'] = 0.025

    model = SPiT(**kwargs) # type: ignore

    if pretrained:
        sd = _get_pretrained_weights('RViT_B16', grad)
        model.load_state_dict(sd)
        return model.eval()

    return model.eval()

def load_RViT_B16(grad:bool=False, pretrained=False) -> SPiT:
    kwargs = {**_architecture_cfg['B'], **_std_cfg}
    kwargs['num_bins'] = 16
    kwargs['tokenizer'] = 'default'
    kwargs['mode'] = 'bilinear'
    kwargs['prvt'] = True
    kwargs['drop_delta'] = not grad
    kwargs['sigma2d'] = 0.025

    model = SPiT(**kwargs) # type: ignore

    if pretrained:
        sd = _get_pretrained_weights('RViT_B16', grad)
        model.load_state_dict(sd)
        return model.eval()

    return model.eval()
