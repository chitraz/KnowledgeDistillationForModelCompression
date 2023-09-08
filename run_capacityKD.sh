# Experiments for section 5.2 in report.

# distil ResNet-14 student from teachers of different depth/capacity on Cifar-100. 

####################################################################################################
# (teacher) ResNet-N -> ResNet-14 (student) on CIFAR-100 using vanilla KD
####################################################################################################

# ResNet-20 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_KD -dataset CIFAR-100 -mode KD_Hinton -s_model ResNet-14 -t_model ResNet-20 -weight saves/ResNet-20_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:69.18_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 0.9 -T 4

# ResNet-32 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_KD -dataset CIFAR-100 -mode KD_Hinton -s_model ResNet-14 -t_model ResNet-32 -weight saves/ResNet-32_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:71.28_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 0.9 -T 4

# ResNet-56 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_KD -dataset CIFAR-100 -mode KD_Hinton -s_model ResNet-14 -t_model ResNet-56 -weight saves/ResNet-56_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:72.34_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 0.9 -T 4

# ResNet-110 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_KD -dataset CIFAR-100 -mode KD_Hinton -s_model ResNet-14 -t_model ResNet-110 -weight saves/ResNet-110_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:74.61_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 0.9 -T 4


####################################################################################################
# (teacher) ResNet-N -> ResNet-14 (student) on CIFAR-100 using DIST
####################################################################################################


# ResNet-20 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_DIST -dataset CIFAR-100 -mode DIST -s_model ResNet-14 -t_model ResNet-20 -weight saves/ResNet-20_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:69.18_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 2 -G 2 -T 4

# ResNet-32 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_DIST -dataset CIFAR-100 -mode DIST -s_model ResNet-14 -t_model ResNet-32 -weight saves/ResNet-32_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:71.28_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 2 -G 2 -T 4

# ResNet-56 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_DIST -dataset CIFAR-100 -mode DIST -s_model ResNet-14 -t_model ResNet-56 -weight saves/ResNet-56_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:72.34_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 2 -G 2 -T 4

# ResNet-110 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_DIST -dataset CIFAR-100 -mode DIST -s_model ResNet-14 -t_model ResNet-110 -weight saves/ResNet-110_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:74.61_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 2 -G 2 -T 4

####################################################################################################
# (teacher) ResNet-N -> ResNet-14 (student) on CIFAR-100 using SRD
####################################################################################################


# ResNet-20 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_SRD -dataset CIFAR-100 -mode SRD -s_model ResNet-14 -t_model ResNet-20 -weight saves/ResNet-20_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:69.18_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 2

# ResNet-32 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_SRD -dataset CIFAR-100 -mode SRD -s_model ResNet-14 -t_model ResNet-32 -weight saves/ResNet-32_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:71.28_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 2

# ResNet-56 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_SRD -dataset CIFAR-100 -mode SRD -s_model ResNet-14 -t_model ResNet-56 -weight saves/ResNet-56_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:72.34_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 2

# ResNet-110 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_SRD -dataset CIFAR-100 -mode SRD -s_model ResNet-14 -t_model ResNet-110 -weight saves/ResNet-110_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:74.61_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 2

####################################################################################################
# (teacher) ResNet-N -> ResNet-14 (student) on CIFAR-100 using SRDwithDIST
####################################################################################################


# ResNet-20 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_SRDwithDIST -dataset CIFAR-100 -mode SRDwithDIST -s_model ResNet-14 -t_model ResNet-20 -weight saves/ResNet-20_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:69.18_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 10

# ResNet-32 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_SRDwithDIST -dataset CIFAR-100 -mode SRDwithDIST -s_model ResNet-14 -t_model ResNet-32 -weight saves/ResNet-32_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:71.28_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 10

# ResNet-56 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_SRDwithDIST -dataset CIFAR-100 -mode SRDwithDIST -s_model ResNet-14 -t_model ResNet-56 -weight saves/ResNet-56_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:72.34_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 10

# ResNet-110 (Teacher) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_SRDwithDIST -dataset CIFAR-100 -mode SRDwithDIST -s_model ResNet-14 -t_model ResNet-110 -weight saves/ResNet-110_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:74.61_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 1 -B 10

####################################################################################################
# Try using an intermediate model (teacher assistant). 
# (T) ResNet-110 → (TA) ResNet-32 → (S) ResNet-14 on CIFAR-100 using vanilla KD. 
# (doesn't work! why? need more steps? maybe try (T)R-110 -> (TA1)R-56 -> (TA2)R-32 -> (S)R-14 )
####################################################################################################

# ResNet-110 (Teacher) -> ResNet-32 (Teacher Assistant)
python scripts/Distiller.py -name capacity_TAKD -dataset CIFAR-100 -mode KD_Hinton -s_model ResNet-32 -t_model ResNet-110 -weight saves/ResNet-110_from-scratch_CIFAR-100_pretrains_350epochs_ValAcc:74.61_2023-08-01.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 0.9 -T 4

# ResNet-32 (Teacher Assistant) -> ResNet-14 (Student)
python scripts/Distiller.py -name capacity_TAKD -dataset CIFAR-100 -mode KD_Hinton -s_model ResNet-14 -t_model ResNet-32 -weight saves/ResNet-32_KD_Hinton_ResNet-110_CIFAR-100_capacity_TAKD_350epochs_ValAcc:72.64_2023-09-04.pth -epochs 350 -train_bs 128 -w_decay 0.0005 -momentum 0.9 -lr 0.1 -gamma 0.1 -step_s 150 250 320 -A 0.9 -T 4
