
<p align="center">
  <img src="./figs/KVQE Challenge.png" alt="image" style="width:1000px;">
</p>

# KVQE: Kwai Video Quality Assessment and Enhancement for Short-form Videos

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)]()  [![Project](https://img.shields.io/badge/Project-Page-blue.svg)]() [![Data](https://img.shields.io/badge/Dataset-Link-magenta.svg)](https://drive.google.com/drive/folders/1dkC4NsxMrd6Rxm1IogKe71U8bYy76ojV?usp=sharing)
[![Challenge-VQA](https://img.shields.io/badge/Competition-Codalab-purple.svg)](https://codalab.lisn.upsaclay.fr/competitions/21335) 
[![Challenge-SR](https://img.shields.io/badge/Competition-Codalab-purple.svg)](https://codalab.lisn.upsaclay.fr/competitions/21346) 
![visitors](https://visitor-badge.laobi.icu/badge?page_id=lixinustc/KVQE-Challenge-CVPR-NTIRE2025)
## :bookmark: News!!!
- [x] 2025-02-07: **Two tracks of this competition have been started**
- [x] 2025-02-08: **The first track can refer to the [KVQ competition](https://github.com/lixinustc/KVQ-Challenge-CVPR-NTIRE2024) in the CVPR NTIRE 2024** 

## 📌 Dataset for KVQE 

##  :tada: Challenge Description
This competition has two tracks:

### [Efficient Short-form UGC Video Quality Assessment](https://codalab.lisn.upsaclay.fr/competitions/21335)

The first track is efficient short-form UGC video quality assessment, and the second track is Diffusion-based Super-resolution for the Short-form UGC Images in the Wild. The first track utilizes the KVQ, i.e., the large-scale Kaleidoscope short Video database for Quality assessment, for training, and evaluation. The KVQ database compromises 600 user-uploaded short videos and 3600 processed videos through diverse practical processing workflows. Moreover, it contains nine primary content scenarios in the practical short-form video platform, including landscape, crowd, person, food, portrait, computer graphic (termed as CG), caption, and stage, covering almost all existing creation modes and scenarios, and the ratio of each category of content satisfies the practical online statistics. The quality score of each short-form video and the partial ranked score are annotated by professional researchers on image processing. 

### [Diffusion-based Image Super-resolution for Short-form UGC images in the wild](https://codalab.lisn.upsaclay.fr/competitions/21346)

The second track collected 1800 synthetic paired images with a simulation strategy from the real-world Kwai Platform and 1900 real-world in-the-wild images with only low-quality images. The contents are from the same source as the KVQ datasets. The purpose is to improve the perceptual quality of images in the wild while maintaining the generalization capability. It is encouraged to utilize the diffusion models for methods. Other methods are also welcomed.

This competition aims to establish a new and applicable benchmark for short-form User Generated Content (UGC) quality assessment and enhancement. We are looking forward to the collaborative efforts of our participants, aiming to elevate the quality of short-form video content. The first track will introduce an innovative evaluation methodology that incorporates a coarse-grained quality score measurement, along with fine-grained rankings for more challenging samples. The second track will introduce the user study and non-reference metrics for evaluation.

The top-ranked participants will be awarded by KuaiShou Company and invited to follow the CVPR submission guide for workshops to describe their solution and to submit to the associated NTIRE workshop at CVPR 2025.


## Cite US
Please cite us if this work is helpful to you.
```
@inproceedings{lu2024kvq,
  title={KVQ: Kwai Video Quality Assessment for Short-form Videos},
  author={Lu, Yiting and Li, Xin and Pei, Yajing and Yuan, Kun and Xie, Qizhi and Qu, Yunpeng and Sun, Ming and Zhou, Chao and Chen, Zhibo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

```
@inproceedings{li2024ntire,
  title={NTIRE 2024 Challenge on Short-form UGC Video Quality Assessment: Methods and Results},
  author={Li, Xin and Yuan, Kun and Pei, Yajing and Lu, Yiting and Sun, Ming and Zhou, Chao and Chen, Zhibo and Timofte, Radu and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year={2024}
}
```

## Acknowledgments
The basic code is partially from the below repos.
- [SimpleVQA](https://github.com/sunwei925/SimpleVQA)
- [Dover](https://github.com/VQAssessment/DOVER)
