# Experiments for section 5.3 in report.

# Evaluate KD methods on various T-S arch. 

####################################################################################################
# (depth compression): (T)WRN-40-4 -> (S)WRN-16-4 on CIFAR-100 using various KD methods
####################################################################################################

# DONE in 'run_test.sh', don't have to repeat

# KD
#python scripts/Distiller.py -name compare_KD -dataset CIFAR-100 -mode KD_Hinton -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4

# FitNet
#python scripts/Distiller.py -name compare_FitNet -dataset CIFAR-100 -mode FitNet-like -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -B 2 -hint 3

# AT 
#python scripts/Distiller.py -name compare_AT -dataset CIFAR-100 -mode AT -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -B 1000

# DML (WRN-40-4 <-> WRN-16-4)
#python scripts/Distiller.py -name compare_DML -dataset CIFAR-100 -mode DML -s_model WRN-16-4 -t_model WRN-16-4 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 1

# SRD
#python scripts/Distiller.py -name compare_SRD -dataset CIFAR-100 -mode SRD -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2

# DIST
#python scripts/Distiller.py -name compare_DIST -dataset CIFAR-100 -mode DIST -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2 -G 2 -T 4

# SRDwithDIST
#python scripts/Distiller.py -name compare_SRDwithDIST -dataset CIFAR-100 -mode SRDwithDIST -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 4 -G 10



####################################################################################################
# (width compression):  (T)WRN-40-2 -> (S)WRN-40-1 on CIFAR-100 using various KD methods
####################################################################################################

# KD
python scripts/Distiller.py -name compare_KD -dataset CIFAR-100 -mode KD_Hinton -s_model WRN-40-1 -t_model WRN-40-2 -weight saves/WRN-40-2_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:76.68_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4

# FitNet 
python scripts/Distiller.py -name compare_FitNet -dataset CIFAR-100 -mode FitNet-like -s_model WRN-40-1 -t_model WRN-40-2 -weight saves/WRN-40-2_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:76.68_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -B 2 -hint 3

# AT (+ KD)
python scripts/Distiller.py -name compare_AT -dataset CIFAR-100 -mode AT -s_model WRN-40-1 -t_model WRN-40-2 -weight saves/WRN-40-2_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:76.68_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -B 1000

# DML
python scripts/Distiller.py -name compare_DML -dataset CIFAR-100 -mode DML -s_model WRN-40-1 -t_model WRN-40-2 -weight saves/WRN-40-2_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:76.68_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 1

# SRD
python scripts/Distiller.py -name compare_SRD -dataset CIFAR-100 -mode SRD -s_model WRN-40-1 -t_model WRN-40-2 -weight saves/WRN-40-2_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:76.68_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2

# DIST 
python scripts/Distiller.py -name compare_DIST -dataset CIFAR-100 -mode DIST -s_model WRN-40-1 -t_model WRN-40-2 -weight saves/WRN-40-2_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:76.68_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2 -G 2 -T 4

# SRDwithDIST
python scripts/Distiller.py -name compare_SRDwithDIST -dataset CIFAR-100 -mode SRDwithDIST -s_model WRN-40-1 -t_model WRN-40-2 -weight saves/WRN-40-2_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:76.68_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 10

####################################################################################################
# (depth + width compression): (T)WRN-40-4 -> (S)WRN-16-2 on CIFAR-100 using various KD methods
####################################################################################################

# KD
python scripts/Distiller.py -name compare_KD -dataset CIFAR-100 -mode KD_Hinton -s_model WRN-16-2 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4

# FitNet
python scripts/Distiller.py -name compare_FitNet -dataset CIFAR-100 -mode FitNet-like -s_model WRN-16-2 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -B 2 -hint 3

# AT 
python scripts/Distiller.py -name compare_AT -dataset CIFAR-100 -mode AT -s_model WRN-16-2 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -B 1000

# DML 
python scripts/Distiller.py -name compare_DML -dataset CIFAR-100 -mode DML -s_model WRN-16-2 -t_model WRN-40-4 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 1

# SRD
python scripts/Distiller.py -name compare_SRD -dataset CIFAR-100 -mode SRD -s_model WRN-16-2 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2

# DIST
python scripts/Distiller.py -name compare_DIST -dataset CIFAR-100 -mode DIST -s_model WRN-16-2 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2 -G 2 -T 4

# SRDwithDIST
python scripts/Distiller.py -name compare_SRDwithDIST -dataset CIFAR-100 -mode SRDwithDIST -s_model WRN-16-2 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 4 -G 10

####################################################################################################
# (differnet arch type): (T)WRN-40-4 -> (S)MobileNetV1(x1_0) on CIFAR-100 using various KD methods
####################################################################################################

# KD
python scripts/Distiller.py -name compare_KD -dataset CIFAR-100 -mode KD_Hinton -s_model MobileNetV2_x1_0 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.01 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4

# FitNet
python scripts/Distiller.py -name compare_FitNet -dataset CIFAR-100 -mode FitNet-like -s_model MobileNetV2_x1_0 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.01 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -B 2 -hint 3

# DML
python scripts/Distiller.py -name compare_DML -dataset CIFAR-100 -mode DML -s_model MobileNetV2_x1_0 -t_model WRN-40-4 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.01 -gamma 0.1 -step_s 150 180 210 -A 1 -B 1

# DIST
python scripts/Distiller.py -name compare_DIST -dataset CIFAR-100 -mode DIST -s_model MobileNetV2_x1_0 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.01 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2 -G 2 -T 4

####################################################################################################
# (differnet arch type): (T)ResNet-18 -> (S)MobileNetV1(x1_0) on CIFAR-100 using various KD methods
####################################################################################################

# KD
python scripts/Distiller.py -name compare_KD -dataset CIFAR-100 -mode KD_Hinton -s_model MobileNetV2_x1_0 -t_model ResNet-18 -weight saves/ResNet-18_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:78.13_2023-08-22.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.01 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4

# FitNet
python scripts/Distiller.py -name compare_FitNet -dataset CIFAR-100 -mode FitNet-like -s_model MobileNetV2_x1_0 -t_model ResNet-18 -weight saves/ResNet-18_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:78.13_2023-08-22.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.01 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -B 2 -hint 3

# DML
python scripts/Distiller.py -name compare_DML -dataset CIFAR-100 -mode DML -s_model MobileNetV2_x1_0 -t_model ResNet-18 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.01 -gamma 0.1 -step_s 150 180 210 -A 1 -B 1

# DIST
python scripts/Distiller.py -name compare_DIST -dataset CIFAR-100 -mode DIST -s_model MobileNetV2_x1_0 -t_model ResNet-18 -weight saves/ResNet-18_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:78.13_2023-08-22.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.01 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2 -G 2 -T 4