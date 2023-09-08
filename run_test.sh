# Experiments for section 4.2 in report.

# teacher:WRN-40-4 and student: WRN-16-4 for test.

####################################################################################################
#  Test KD 
####################################################################################################
# KD with alpha = 0.9 and T = 4
python scripts/Distiller.py -name test_KD -dataset CIFAR-100 -mode KD_Hinton -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4

####################################################################################################
#  Test FitNet 
####################################################################################################

# WRN-40-4 -> WRN-16-4 on cifar100 using FitNet (2-stages: hint based training followed by vanilla KD)
python scripts/Distiller.py -name test_FitNet -dataset CIFAR-100 -mode FitNet -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -hint 3

# WRN-40-4 -> WRN-16-4 on cifar100 using FitNet (1-stage: min L_student = (1-A)*L_cls + A*L_kd + B*L_hint )
python scripts/Distiller.py -name test_FitNet -dataset CIFAR-100 -mode FitNet-like -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -B 2 -hint 3


####################################################################################################
#  Test AT
####################################################################################################

# AT (with KD) using alpha = 0.9, T = 4 and beta = 1000
python scripts/Distiller.py -name test_AT -dataset CIFAR-100 -mode AT -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -B 1000

# AT (without KD) using alpha = 0 and beta = 1000
python scripts/Distiller.py -name test_ATnoKD -dataset CIFAR-100 -mode AT -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0 -T 4 -B 1000


####################################################################################################
#  Test DML
####################################################################################################

# WRN-16-4 <-> WRN-16-4 on CIFAR-100 using DML (same teacher and student)
python scripts/Distiller.py -name testDML -dataset CIFAR-100 -mode DML -s_model WRN-16-4 -t_model WRN-16-4 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 1

# WRN-40-4 <-> WRN-16-4 on CIFAR-100 using DML (see if having a stonger model in the cohort improve performace. yes slightly)
python scripts/Distiller.py -name testDML -dataset CIFAR-100 -mode DML -s_model WRN-16-4 -t_model WRN-40-4 -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 1

####################################################################################################
#  Test DIST
####################################################################################################

# DIST using alpha = 1, beta = 2 , gamma = 2 and T = 4
python scripts/Distiller.py -name testDIST -dataset CIFAR-100 -mode DIST -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2 -G 2 -T 4

####################################################################################################
#  Test SRD with different SRD loss design: mse (defualt), pmse, KL and DIST. (loss balance values?)
####################################################################################################

# SRD with alpha = 1, beta = 2
python scripts/Distiller.py -name testSRD -dataset CIFAR-100 -mode SRD -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2

# SRDwithDIST with alpha = 1, beta = 10 (L_intra and L_inter balanced) [new results?]
python scripts/Distiller.py -name testSRDwithDIST -dataset CIFAR-100 -mode SRDwithDIST -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 10


####### other SRD loss design from SRD paper. ###### 
# weight balance? 

# SRDwithPMSE with alpha = 1, beta = 2 
python scripts/Distiller.py -name testSRDwithPMSE -dataset CIFAR-100 -mode SRDwithPMSE -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2

# SRDwithKL with alpha = 1, beta = 2
python scripts/Distiller.py -name testSRDwithKL -dataset CIFAR-100 -mode SRDwithKL -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2