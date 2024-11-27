<div align="center">

# A Spitting Image: Modular Superpixel Tokenization in Vision Transformers

**[Marius Aasan](https://www.mn.uio.no/ifi/english/people/aca/mariuaas/), [Odd Kolbjørnsen](https://www.mn.uio.no/math/english/people/aca/oddkol/), [Anne Schistad Solberg](https://www.mn.uio.no/ifi/english/people/aca/anne/), [Adín Ramírez Rivera](https://www.mn.uio.no/ifi/english/people/aca/adinr/)** <br>


**[DSB @ IFI @ UiO](https://www.mn.uio.no/ifi/english/research/groups/dsb/)** <br>

[![Website](https://img.shields.io/badge/Website-green)](https://dsb-ifi.github.io/SPiT/)
[![PaperArxiv](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2408.07680)
[![PaperECCVW](https://img.shields.io/badge/Paper-ECCVW_2024-blue)](https://sites.google.com/view/melex2024)
[![NotebookExample](https://img.shields.io/badge/Notebook-Example-orange)](https://nbviewer.jupyter.org/github/dsb-ifi/SPiT/blob/main/notebooks/eval_in1k.ipynb) <br>

![SPiT Figure 1](/assets/fig1.png#gh-light-mode-only "Examples of feature maps from SPiT-B16")
![SPiT Figure 1](/assets/fig1_dark.png#gh-dark-mode-only "Examples of feature maps from SPiT-B16")

</div>

## SPiT: Superpixel Transformers

This repo contains code and weights for **A Spitting Image: Modular Superpixel Tokenization in Vision Transformers**, accepted for MELEX, ECCVW 2024.

For an introduction to our work, visit the [project webpage](https://dsb-ifi.github.io/SPiT/). 

## Installation

We are working on releasing this package on PyPi, however, the package can currently be installed via:

```bash
# HTTPS
pip install git+https://github.com/dsb-ifi/SPiT.git

# SSH
pip install git+ssh://git@github.com/dsb-ifi/SPiT.git
```

## Loading models

To load a Superpixel Transformer model, we suggest using the wrapper:

```python
from spit import load_model

model = load_model.load_SPiT_B16(grad=True, pretrained=True)
```

This will load the model and downloaded the pretrained weights, stored in your local `torch.hub` directory. If you would rather download the full weights, please use:

| Model | Link | MD5 |
|-|-|-|
| SPiT-S16 | [Manual Download](https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EZ57Sad2uf9Dizwm3VYhvw4BVdHOxsEJcgyf4vgKsdmgZg) |8e899c846a75c51e1c18538db92efddf|
| SPiT-S16 (w. grad.) | [Manual Download](https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/Eb9FViSwap5JqYe1mtlC3jQBE-nAMG88MfJfmypT_J8r0Q) |e49be7009c639c0ccda4bd68ed34e5af|
| SPiT-B16 | [Manual Download](https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EXhsshO-DvlIii87kyyEVtoBRFbZaTp8SqTgDJhQ1iQIBw) |9d3483a4c6fdaf603ee6528824d48803|
| SPiT-B16 (w. grad.) | [Manual Download](https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EcahlrAzXZ5Bsozrqs4dWLABHFX-V5VH8jQR5ygHhZH30A) |9394072a5d488977b1af05c02aa0d13c|
| ViT-S16 | [Manual Download](https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EWqHDQvY5V5PjKkMmO5fcFEBKuN6WTfr4a99u8vpNT67WQ) |73af132e4bb1405b510a5eb2ea74cf22|
| ViT-S16 (w. grad.)    | [Manual Download](https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EenEECYQaQZFl_GeU2N9q7YB-XOHNyaJXHnC74qREU3cSQ) |b8e4f1f219c3baef47fc465eaef9e0d4|
| ViT-B16 | [Manual Download](https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EUWJM_RY9IRPvM9dsp2Zzi8B6ZOnhQ_C666TMESzmAQ0sQ) |ce45dcbec70d61d1c9f944e1899247f1|
| ViT-B16 (w. grad.)    | [Manual Download](https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EdGx5GaXRshPpOh0gsCHU4cBeZ0FxexzuBm7vTtm67nuTw) |1caa683ecd885347208b0db58118bf40|
|RViT-S16| Coming Soon | |
| RViT-S16 (w. grad.) | Coming Soon | |
| RViT-B16 | [Manual Download](https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/Ed9R0bQOmslLiPnFX_P0hRoBUf_zQ4pfHXZ3BpQ4iW8JYA) |18c13af67d10f407c3321eb1ca5eb568|
| RViT-B16 (w. grad.) | [Manual Download](https://uio-my.sharepoint.com/:u:/g/personal/mariuaas_uio_no/EflpV7TP04RKmxg1qfiNovUBo149q0P9j4tmoOTQ-NkV-Q) |50d25403adfd5a12d7cb07f7ebfced97|


## More Examples

We provide a [Jupyter notebook](https://nbviewer.jupyter.org/github/dsb-ifi/SPiT/blob/main/notebooks/eval_in1k.ipynb) as a sandbox for loading, evaluating, and extracting segmentations for the models. *Examples will be updated along with new releases and updates for the project repo*.

## Notes:

### RViT and On-Line Voronoi Tesselation

Currently the code features some slight modifications to streamline use of the RViT models. The original RViT models sampled partitions from a dataset of pre-computed Voronoi tesselations for training and evaluation. This is impractical for deployment, and we have yet to implement a CUDA kernel for computing Voronoi with lower memory overhead.

However, we have developed a fast implementation for generating fast tesselations with PCA trees [1], which mimic Voronoi tesselations relatively well, and can be computed on-the-fly. There are, however still some minor issues with the small capacity RViT models. Consequently, the RViT-B16 models will perform marginally different than the reported results in the paper. *We appreciate the readers patience with regard to this matter.*

Note that the RViT models are inherently stochastic so that different runs can yield different results. Also, it is worth mentioning that SPiT models can yield slightly different results for each run, due to nondeterministic behaviours in CUDA kernels.


[1] Refinements to nearest-neighbor searching in $k$-dimensional trees [(Sproull, 1991)](https://doi.org/10.1007/BF01759061)

## Progress and Current Todo's:

- [X] Include foundational code and model weights.
- [X] Add manual links with MD5 hash for manual weight download.
- [X] Add module for loading models, and provide example notebook.
- [X] Create temporary solution to on-line Voronoi tesselation.
- [ ] Add standalone train and eval scripts.
- [ ] Add CUDA kernels for on-line Voronoi Tesselations.
- [ ] Add example for extracting attribution maps with Att.Flow and Proto.PCA.
- [ ] Add example for computing sufficiency and comprehensiveness.
- [ ] Add assets for computed attribution maps for XAI experiments.
- [ ] Add code and examples for salient segmentation.
- [ ] Add code and examples for feature correspondences.

## Citation

If you find our work useful, please consider citing our work.
```
@inproceedings{Aasan2024,
  title={A Spitting Image: Modular Superpixel Tokenization in Vision Transformers},
  author={Aasan, Marius and Kolbj\o{}rnsen, Odd and Schistad Solberg, Anne and Ram\'irez Rivera, Ad\'in},
  booktitle={{CVF/ECCV} More Exploration, Less Exploitation Workshop ({MELEX} {ECCVW})},
  year={2024}
}
```
