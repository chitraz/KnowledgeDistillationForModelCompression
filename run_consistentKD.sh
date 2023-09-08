# Experiments for section 5.1 in report

####################################################################################################
# Check the effect of teacher and student seeing differnt view of traninig image.
####################################################################################################

# vanilla KD (same view, consistent teacher) 
python scripts/Distiller.py -name consistent_KD -dataset CIFAR-100 -mode KD_Hinton -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4

# vanilla KD (different view due to pre-computing, in-consistent teacher) 
python scripts/Distiller.py -name inconsistent_KD -dataset CIFAR-100 -mode KD_Hinton -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 0.9 -T 4 -pre_comp

# DIST (same view, consistent teacher) 
python scripts/Distiller.py -name consistent_DIST -dataset CIFAR-100 -mode DIST -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2 -G 2 -T 4

# DIST (different view, in-consistent teacher) 
python scripts/Distiller.py -name inconsistent_DIST -dataset CIFAR-100 -mode DIST -s_model WRN-16-4 -t_model WRN-40-4 -weight saves/WRN-40-4_from-scratch_CIFAR-100_pretrains_240epochs_ValAcc:79.16_2023-07-28.pth -epochs 240 -train_bs 64 -w_decay 0.0005 -momentum 0.9 -lr 0.05 -gamma 0.1 -step_s 150 180 210 -A 1 -B 2 -G 2 -T 4 -pre_comp
