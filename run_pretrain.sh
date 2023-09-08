# Experiments for section 4.1 in report

# ResNets-8 and ResNet-32 on cifar10 using training setting from SRD paper. 
python scripts/Distiller.py -name pretrains -dataset CIFAR-10 -mode from-scratch -s_model ResNet-8 -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320
python scripts/Distiller.py -name pretrains -dataset CIFAR-10 -mode from-scratch -s_model ResNet-32 -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320

# ResNets-8, 14, .., 110 on cifar100 using training setting from SRD paper.
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model ResNet-8 -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model ResNet-14 -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model ResNet-20 -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model ResNet-32 -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model ResNet-44 -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model ResNet-56 -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model ResNet-110 -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320

# ResNet-18, ResNet-50 (resnetv2.py, other more common resnet variant) on cifar100 
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model ResNet-18 -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model ResNet-50 -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320

# WRNs on cifar100 using training settings from DIST paper
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model WRN-16-1 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model WRN-16-2 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model WRN-16-4 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model WRN-40-1 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model WRN-40-2 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model WRN-40-4 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210

# MobileNet-V2 on cifar100 using training settings from DIST paper
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model MobileNetV2_x1_0 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.01 -gamma 0.1 -step_s 150 180 210
python scripts/Distiller.py -name pretrains -dataset CIFAR-100 -mode from-scratch -s_model MobileNetV2_x0_5 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.01 -gamma 0.1 -step_s 150 180 210