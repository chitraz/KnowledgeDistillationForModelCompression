import torch.nn as nn
from torchvision import models

# resnets for cifar 
from models.resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110

# resnets for ImageNet
from models.resnetv2 import ResNet18, ResNet34, ResNet50,ResNet101, ResNet152

# wide resnet 
from models.wrn import wrn_16_1, wrn_16_2, wrn_16_4, wrn_40_1, wrn_40_2, wrn_40_4

# GhostNet
from models.ghostnet import ghostnet

# MobileNetV2
from models.mobilenetv2 import MobileNet_v2_x0_5, MobileNet_v2_x1_0

def get_Model(model_name, NUM_CLASS):
    if model_name == 'ResNet-8':
        model = resnet8(num_classes=NUM_CLASS)
        return model
    elif model_name == 'ResNet-14':
        model = resnet14(num_classes=NUM_CLASS)
        return model
    elif model_name == 'ResNet-20':
        model = resnet20(num_classes=NUM_CLASS)
        return model
    elif model_name == 'ResNet-32':
        model = resnet32(num_classes=NUM_CLASS)
        return model
    elif model_name == 'ResNet-44':
        model = resnet44(num_classes=NUM_CLASS)
        return model
    elif model_name == 'ResNet-56':
        model = resnet56(num_classes=NUM_CLASS)
        return model    
    elif model_name == 'ResNet-110':
        model = resnet110(num_classes=NUM_CLASS)
        return model
    
    elif model_name == 'ResNet-18':
        model = ResNet18(num_classes=NUM_CLASS)
        return model
    elif model_name == 'ResNet-34':
        model = ResNet34(num_classes=NUM_CLASS)
        return model
    elif model_name == 'ResNet-50':
        model = ResNet50(num_classes=NUM_CLASS)
        return model
    elif model_name == 'ResNet-101':
        model = ResNet101(num_classes=NUM_CLASS)
        return model
    elif model_name == 'ResNet-152':
        model = ResNet152(num_classes=NUM_CLASS)
        return model
    
    # note: drop out in each res block is 0 by defualt. drop out disabled. 
    elif model_name == 'WRN-40-4':
        model = wrn_40_4(num_classes=NUM_CLASS)
        return model 
    elif model_name == 'WRN-40-2':
        model = wrn_40_2(num_classes=NUM_CLASS)
        return model
    elif model_name == 'WRN-40-1':
        model = wrn_40_1(num_classes=NUM_CLASS)
        return model
    elif model_name == 'WRN-16-4':
        model = wrn_16_4(num_classes=NUM_CLASS)
        return model 
    elif model_name == 'WRN-16-2':
        model = wrn_16_2(num_classes=NUM_CLASS)
        return model
    elif model_name == 'WRN-16-1':
        model = wrn_16_1(num_classes=NUM_CLASS)
        return model 

    elif model_name == 'MobileNetV2_x1_0':
        model = MobileNet_v2_x1_0(num_classes=NUM_CLASS)
        return model 
    elif model_name == 'MobileNetV2_x0_5':
        model = MobileNet_v2_x0_5(num_classes=NUM_CLASS)
        return model 
    elif model_name == 'GhostNet_x1_0':
        model = ghostnet(num_classes=NUM_CLASS)
        return model 
    
    else:
        return None