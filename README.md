# Knowledge Distillation for Model Compression

This repo contains code written for my [MSc Project](https://github.com/chitraz/KnowledgeDistillationForModelCompression/files/15062925/FinalReport_Chitra.pdf). More specfically: 
  - PyTorch Implementation ([Distiller.py](scripts/Distiller.py), [Utils.py](scripts/Utils.py), [Dataset.py](scripts/Dataset.py), [Models.py](scripts/Models.py), [KD_methods.py](scripts/KD_methods.py)) to conduct knowledge distillation experiemnts on CIFAR-10/100 or ImageNet-1k using various residual CNN networks. 
  - Shell Scripts to run all the experiments conducted and Juperter Noteboks for visualisations
    



Where various KD methods are explored to distill using primaly residual CNNs models on CIFAR-10/100 dataset. In addition a 



## SRDwithDIST Framework
This is a simple modification made to the [SRD](https://arxiv.org/abs/2205.06701) method where instead of the mse(), between the teacher's logits and cross-network logits, a correlation based loss, following [DIST](https://arxiv.org/abs/2205.10536), is used to relax the matching. Given a batch of teacher logits and cross-network logits, they are matched row-wise and column-wise using the pearson correlation coefficient in order to align their relative rankings.   

<img src="https://github.com/chitraz/KnowledgeDistillationForModelCompression/assets/40371968/61d02532-9403-4e64-bdd8-ac4555614c64" width="1000" />

## Main results on CIFAR-100 

Shows top-1 classification accuracies on CIFAR-100. See [run_compareKD.sh](run_compareKD.sh) for the commands used. 

| Teacher Architecture <br> [#parameters] <br> Student Architecture <br> [#parameters] | WRN-40-4 <br> [8.97M] <br> WRN-16-4 <br> [2.77M] | WRN-40-2 <br> [2.26M] <br> WRN-40-1 <br> [0.57M] | WRN-40-4 <br> [8.97M] <br> WRN-16-2 <br> [0.70M] | WRN-40-4 <br> [8.97M] <br> MobileNet-V2 <br> [2.24M]| ResNet-18 <br> [11.22M] <br> MobileNet-V2 <br> [2.24M]|
| :------------- | :-----: | :-----: | :-----: | :-----: | :-----: |
| Teacher | 79.16 | 76.68 | 79.16 | 79.16 | 78.13 |
| Student     | 76.91 | 71.3  | 73.45 | 69.66 | 69.66 |
| KD [[paper](https://arxiv.org/abs/1503.02531)]         | 78.65 (+1.74) | 73.56 (+2.26) | 75.01 (+1.56) | 72.93 (+3.27) | 73.40 (+3.74)  |
| FitNet [[paper](https://arxiv.org/abs/1412.6550)]      | 79.15 (+2.24) | 74.11 (+2.81) | 74.66 (+1.21) | 73.84 (+4.18) | 73.19 (+3.53) |
| AT [[paper](https://arxiv.org/abs/1612.03928), [GitHub](https://github.com/szagoruyko/attention-transfer)]          | 79.05 (+2.14) | 73.90 (+2.60)  | 74.38 (+0.93) | \-    | \-    |
| DML [[paper](https://arxiv.org/abs/1706.00384)]         | 78.69 (+1.78) | 73.72 (+2.42) | 74.76 (+1.31) | 72.20 (+2.54)  | 72.26 (+2.60) |
| DIST [[paper](https://arxiv.org/abs/2205.10536), [GitHub](https://github.com/hunto/DIST_KD)]        | 79.43 (+2.52) | 74.44 (+3.14) | 75.50 (+2.05)  | 73.44 (+3.78) | 72.68 (+3.02) |
| SRD [[paper](https://arxiv.org/abs/2205.06701), [GitHub](https://github.com/jingyang2017/SRD_ossl)]         | 79.53 (+2.62) | **74.67 (+3.37)** | 75.94 (+2.49) | \-    | \-    |
| SRDwithDIST | **80.39 (+3.48)** | 74.43 (+3.13) | **76.19 (+2.74)** | \-    | \-    |

*missing some entries due to implementation limitation: currently using a simple 1x1 conv adaptor to handel teacher and student shape mismatch for thoses methods so can only handel 
 

## Installation

### Environment
 The experiments were conducted on a personal PC with: 
 - Ubuntu 22.04
 - Python 3.8
 - PyTorch 2.0
 - CPU: Ryzen 9 5900x
 - GPU: Nvidia RTX 3090

### Example


## Future works  
  - Get results on ImageNet to see how the method scale
  - Explore KD alongside other orthogonal compression techniques: Pruning/Quantisation 
  - Explore KD for compressing Object Detection models
