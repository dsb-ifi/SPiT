---
layout: project_page
permalink: /

title: A Spitting Image: Modular Superpixel Tokenization in Vision Transformers
authors:
    Marius Aasan, Odd Kolbjørnsen, Anne Schistad Solberg, Adín Ramirez Rivera
affiliations:
    University of Oslo
    Aker BP
    SFI Visual Intelligence
paper: https://arxiv.org
# video: https://www.youtube.com/@UniOslo
code: https://github.com/dsb-ifi/SPiT
# data: https://huggingface.co/docs/

abstract: Vision Transformer (ViT) architectures traditionally employ a grid-based approach to tokenization independent of the semantic content of an image. We propose a modular superpixel tokenization strategy which decouples tokenization and feature extraction; a shift from contemporary approaches where these are treated as an undifferentiated whole. Using on-line content-aware tokenization and scale- and shape-invariant positional embeddings, we perform experiments and ablations that contrast our approach with patch-based tokenization and randomized partitions as baselines.  We show that our method significantly improves the faithfulness of attributions, gives pixel-level granularity on zero-shot unsupervised dense prediction tasks, while maintaining predictive performance in classification tasks. Our approach provides a modular tokenization framework commensurable with standard architectures, extending the space of ViTs to a larger class of semantically-rich models.

---
## Results


## Citation
```
@inproceedings{Aasan2024,
  title={A Spitting Image: Modular Superpixel Tokenization in Vision Transformers},
  author={Aasan, Marius and Kolbj\ornsen, Odd and Schistad Solber, Anne and Ram\'irez Rivera, Ad\'in},
  boottitle={{CVF/ECCV} More Exploration, Less Exploitation ({MELEX} {ECCVW})},
  year={2024}
}
```
