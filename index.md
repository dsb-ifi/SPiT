---
layout: project_page
permalink: /

title: "A Spitting Image: Modular Superpixel Tokenization in Vision Transformers"
authors: 
  - name: Marius Aasan
    link: https://www.mn.uio.no/ifi/english/people/aca/mariuaas/
    affiliation: 1, 3
  - name: Odd Kolbjørnsen
    link: https://www.mn.uio.no/math/english/people/aca/oddkol/
    affiliation: 2, 3
  - name: Anne Schistad Solberg
    link: https://www.mn.uio.no/ifi/english/people/aca/anne/
    affiliation: 1, 3
  - name: Adín Ramirez Rivera
    link: https://www.mn.uio.no/ifi/english/people/aca/adinr/
    affiliation: 1, 3
affiliations: 
  - name: University of Oslo
    link: https://www.mn.uio.no/ifi/forskning/grupper/dsb/
  - name: Aker BP
  - name: SFI Visual Intelligence  
    link: https://www.visual-intelligence.no/
paper: https://arxiv.org
# video: https://www.youtube.com/@UniOslo
code: https://github.com/dsb-ifi/SPiT
# data: https://huggingface.co/docs/

abstract: Vision Transformer (ViT) architectures traditionally employ a grid-based approach to tokenization independent of the semantic content of an image. We propose a modular superpixel tokenization strategy which decouples tokenization and feature extraction; a shift from contemporary approaches where these are treated as an undifferentiated whole. Using on-line content-aware tokenization and scale- and shape-invariant positional embeddings, we perform experiments and ablations that contrast our approach with patch-based tokenization and randomized partitions as baselines.  We show that our method significantly improves the faithfulness of attributions, gives pixel-level granularity on zero-shot unsupervised dense prediction tasks, while maintaining predictive performance in classification tasks. Our approach provides a modular tokenization framework commensurable with standard architectures, extending the space of ViTs to a larger class of semantically-rich models.

---
## Results


## Citation
{% raw %}
```
@inproceedings{Aasan2024,
  title={A Spitting Image: Modular Superpixel Tokenization in Vision Transformers},
  author={Aasan, Marius and Kolbj\ornsen, Odd and Schistad Solber, Anne and Ram\'irez Rivera, Ad\'in},
  boottitle={{CVF/ECCV} More Exploration, Less Exploitation ({MELEX} {ECCVW})},
  year={2024}
}
```
{% endraw %}