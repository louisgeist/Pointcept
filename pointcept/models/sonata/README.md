# Sonata

> **Note (MALiBU3D release).** Header, citation and license of the upstream Pointcept Sonata README. This
> repository only keeps the Sonata-v1m2 model (`sonata_v1m2_uni_teacher_head.py`); the indoor pre-training and
> probing recipes are in [upstream Pointcept](https://github.com/Pointcept/Pointcept). Sonata pre-training on
> MALiBU3D uses `configs/malibu3d/pretrain-sonata-v1m2-malibu3d.py`.

This repo is the official training codebase of the paper **_Sonata: Self-Supervised Learning of Reliable Point Representations_**. Sonata is a powerful self-supervised learning framework (as well as our pre-trained model) that shows extraordinary parameter efficiency, data efficiency, and fine-tuning accuracy.

*We recommend beginning with our inference [demo](https://github.com/facebookresearch/sonata) and it is also a tiny library for people who want to integrate pre-trained Sonata into their project.*

[ **Pretrain** ] [ **Sonata** ] - [ [Homepage](https://xywu.me/sonata/) ] [ [Paper](https://arxiv.org/abs/2503.16429) ] [ [Inference Demo](https://github.com/facebookresearch/sonata) ] [ [Bib](#citation) ]

<div align='left'>
<img src="https://raw.githubusercontent.com/pointcept/assets/main/sonata/teaser.png" alt="teaser" width="800" />
</div>

## Citation
If you find _Sonata_ useful to your research, please consider citing our line of works as an acknowledgment. (੭ˊ꒳​ˋ)੭✧
```bib
@inproceedings{wu2025sonata,
    title={Sonata: Self-Supervised Learning of Reliable Point Representations},
    author={Wu, Xiaoyang and DeTone, Daniel and Frost, Duncan and Shen, Tianwei and Xie, Chris and Yang, Nan and Engel, Jakob and Newcombe, Richard and Zhao, Hengshuang and Straub, Julian},
    booktitle={CVPR},
    year={2025}
}
```

```bib
@inproceedings{wu2024ptv3,
    title={Point Transformer V3: Simpler, Faster, Stronger},
    author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}
```
```bib
@inproceedings{wu2024ppt,
    title={Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training},
    author={Wu, Xiaoyang and Tian, Zhuotao and Wen, Xin and Peng, Bohao and Liu, Xihui and Yu, Kaicheng and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}
```
```bib
@inproceedings{wu2023masked,
  title={Masked Scene Contrast: A Scalable Framework for Unsupervised 3D Representation Learning},
  author={Wu, Xiaoyang and Wen, Xin and Liu, Xihui and Zhao, Hengshuang},
  journal={CVPR},
  year={2023}
}
```
```bib
@inproceedings{wu2022ptv2,
    title={Point transformer V2: Grouped Vector Attention and Partition-based Pooling},
    author={Wu, Xiaoyang and Lao, Yixing and Jiang, Li and Liu, Xihui and Zhao, Hengshuang},
    booktitle={NeurIPS},
    year={2022}
}
```
```bib
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished={\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

## License
- Sonata code is released by Meta under the [Apache 2.0 license](https://github.com/facebookresearch/sonata/blob/main/LICENSE),
- Sonata weight is released under the [CC-BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
(restricted by NC of datasets like HM3D, ArkitScenes).
- For commercial usage, please removing datasets restrict by non-commercial license.
