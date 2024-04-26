# Knowledge Distillation for Model Compression

This repo contains code written for my [MSc Project](https://github.com/chitraz/KnowledgeDistillationForModelCompression/files/15062925/FinalReport_Chitra.pdf). More specfically: 
  - PyTorch Implementation ([Distiller.py](scripts/Distiller.py), [Utils.py](scripts/Utils.py), [Dataset.py](scripts/Dataset.py), [Models](scripts/Models.py), [KD_methods.py](scripts/KD_methods.py)) to  
  - Shell Scripts to run the experiments conducted and Juperter Noteboks for visualisations for report
    



Where various KD methods are explored to distill using primaly residual CNNs models on CIFAR-10/100 dataset. In addition a 



## SRDwithDIST Framework

<img src="https://github.com/chitraz/KnowledgeDistillationForModelCompression/assets/40371968/61d02532-9403-4e64-bdd8-ac4555614c64" width="800" />




## Main Results 

| Teacher Architecture <br> [#parameters] <br> Student Architecture <br> [#parameters] | WRN-40-4 <br> [8.97M] <br> WRN-16-4 <br> [2.77M] | WRN-40-2 <br> [2.26M] <br> WRN-40-1 <br> [0.57M] | WRN-40-4 <br> [8.97M] <br> WRN-16-2 <br> [0.70M] | WRN-40-4 <br> [8.97M] <br> MobileNet-V2 <br> [2.24M]| ResNet-18 <br> [11.22M] <br> MobileNet-V2 <br> [2.24M]|
| :------------- | :-----: | :-----: | :-----: | :-----: | :-----: |
| Teacher | 79.16 | 76.68 | 79.16 | 79.16 | 78.13 |
| Student     | 76.91 | 71.3  | 73.45 | 69.66 | 69.66 |
| KD [[paper](https://arxiv.org/abs/1503.02531)]         | 78.65 | 73.56 | 75.01 | 72.93 | 73.4  |
| FitNet [[paper](https://arxiv.org/abs/1412.6550)]      | 79.15 | 74.11 | 74.66 | 73.84 | 73.19 |
| AT [[paper](https://arxiv.org/abs/1612.03928), [GitHub](https://github.com/szagoruyko/attention-transfer)]          | 79.05 | 73.9  | 74.38 | \-    | \-    |
| DML [[paper](https://arxiv.org/abs/1706.00384)]         | 78.69 | 73.72 | 74.76 | 72.2  | 72.26 |
| DIST [[paper](https://arxiv.org/abs/2205.10536), [GitHub](https://github.com/hunto/DIST_KD)]        | 79.43 | 74.44 | 75.5  | 73.44 | 72.68 |
| SRD [[paper](https://arxiv.org/abs/2205.06701), [GitHub](https://github.com/jingyang2017/SRD_ossl)]         | 79.53 | **74.67** | 75.94 | \-    | \-    |
| SRDwithDIST | **80.39** | 74.43 | **76.19** | \-    | \-    |


## Installation

### Requirements


### Example


## Future works  
  - Get results on ImageNet to see how the methods scale
  - Explore KD alongside other compression techniques: Pruning/Quantisation 
  - Explore KD for compressing Object Detection models
