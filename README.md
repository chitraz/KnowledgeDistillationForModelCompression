# Knowledge Distillation for Model Compression

This repo contains the code written to conduct deep learning experiments for my [MSc Project](https://github.com/chitraz/KnowledgeDistillationForModelCompression/files/15062925/FinalReport_Chitra.pdf). Where various KD methods are explored to distill using primaly residual CNNs models on CIFAR-10/100 dataset. In addition a 




  - [Distiller.py](scripts/Distiller.py)
  - [Dataset.py](scripts/Dataset.py)
  - [KD_methods.py](scripts/KD_methods.py)
  - [Models](scripts/Models.py)
  - [Utils.py](scripts/Utils.py)



## Framework

<img src="https://github.com/chitraz/KnowledgeDistillationForModelCompression/assets/40371968/61d02532-9403-4e64-bdd8-ac4555614c64" width="800" />




## Main Results 

| Teacher <br> Student | WRN-40-4 [8.97M] <br> WRN-16-4 [2.77M] | WRN-40-2 [2.26M] <br> WRN-40-1 [0.57M] | WRN-40-4 [8.97M] <br> WRN-16-2 [0.70M] | WRN-40-4 [8.97M] <br> MobileNet-V2 [2.24M]| ResNet-18 [11.22M] <br> MobileNet-V2 [2.24M]|
| :----------- | :-----: | :-----: | :-----: | :-----: | :-----: |
| Teacher | 79.16 | 76.68 | 79.16 | 79.16 | 78.13 |
| Student     | 76.91 | 71.3  | 73.45 | 69.66 | 69.66 |
| KD          | 78.65 | 73.56 | 75.01 | 72.93 | 73.4  |
| FitNet      | 79.15 | 74.11 | 74.66 | 73.84 | 73.19 |
| AT          | 79.05 | 73.9  | 74.38 | \-    | \-    |
| DML         | 78.69 | 73.72 | 74.76 | 72.2  | 72.26 |
| DIST        | 79.43 | 74.44 | 75.5  | 73.44 | 72.68 |
| SRD         | 79.53 | 74.67 | 75.94 | \-    | \-    |
| SRDwithDIST | 80.39 | 74.43 | 76.19 | \-    | \-    |


## Running


## Future works  

  - Expore KD alongside other compression techniques: Pruning/Quantisation 
  - KD for compressing Object Detection models
