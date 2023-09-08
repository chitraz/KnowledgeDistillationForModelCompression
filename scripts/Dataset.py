import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

import os
import cv2
import numpy as np

'''##########################################################################################################################
        CIFAR-100
##########################################################################################################################'''
class cifar100_dataset(datasets.CIFAR100):
    # OVERWRITE the __getitem__ method to also return the image index
    def __getitem__(self, idx):
        # get sample
        image = self.data[idx]
        GT = self.targets[idx]

        # apply image augmentation 
        image = self.transform(image)

        # convert GT label to tensor
        GT = torch.tensor([GT], dtype = torch.long)

        return image, GT, idx

def get_cifar100(DatasetType='train', Dataset_dir = 'dataset'):
    if DatasetType=='train':
        Dataset = cifar100_dataset(Dataset_dir, train=True,
            transform= transforms.Compose(
                    [
                        transforms.ToTensor(), 
                        transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                        transforms.Normalize(mean = [0.5071, 0.4867, 0.4408], std = [0.2675, 0.2565, 0.2761]),
                        transforms.RandomHorizontalFlip(0.5)  # mirror image with 0.5 probability
                    ]
                )
            )
    elif DatasetType=='test':
        Dataset = cifar100_dataset(Dataset_dir, train=False,
            transform=transforms.Compose(
                    [
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = [0.5071, 0.4867, 0.4408], std = [0.2675, 0.2565, 0.2761])
                    ]
                )
            )
    return Dataset

CIFAR100_Fine_labels = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
    "chair","chimpanzee","clock","cloud","cockroach","couch","cra","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo",
    "keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree",
    "motorcycle","mountain","mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree",
    "pear","pickup_truck","pine_tree","plain","plate","poppy","porcupine","possum","rabbit",
    "raccoon","ray","road","rocket","rose","sea","seal","shark","shrew","skunk","skyscraper",
    "snail","snake","spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank",
    "telephone","television","tiger","tractor","train","trout","tulip","turtle","wardrobe",
    "whale","willow_tree","wolf","woman","worm",
]


'''##########################################################################################################################
        CIFAR-10
##########################################################################################################################'''
class cifar10_dataset(datasets.CIFAR10):
    # OVERWRITE the __getitem__ method so that it also return the image index. 
    def __getitem__(self, idx):
        # get sample
        image = self.data[idx]
        GT = self.targets[idx]

        # apply image augmentation 
        image = self.transform(image)

        # convert GT label to tensor
        GT = torch.tensor([GT], dtype = torch.long)

        return image, GT, idx
    
def get_cifar10(DatasetType = 'train', Dataset_dir = 'dataset'):
    if DatasetType=='train':
        Dataset = cifar10_dataset(Dataset_dir, train=True,
            transform= transforms.Compose(
                    [
                        transforms.ToTensor(), 
                        # take random crop of 32x32 from the padded 40x40 image
                        # fill padding with reflected pix values i.e [1,2,3,4] -> [..,2, 1,2,3,4, 3,..]
                        transforms.RandomCrop(32, padding=4, padding_mode='reflect'), 
                        transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010]),
                        transforms.RandomHorizontalFlip(0.5)  # mirror image with 50% probability
                    ]
                )
            )
    elif DatasetType=='test':
        Dataset = cifar10_dataset(Dataset_dir, train=False,
            transform=transforms.Compose(
                    [
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
                    ]
                )
            )
    return Dataset

CIFAR10_labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

'''##########################################################################################################################
        ImageNet-1k  
##########################################################################################################################'''
import sys
# custon dataset class
class ImageNetDataset(Dataset):
    def __init__(self, DatasetType = 'train', Dataset_dir = 'ILSVRC12'):
        
        # list of tuples containing (image path, GT label) 
        self.dataset = []

        # ImageNet's mean and std pixel values 
        self.MEAN_PIX = [0.485, 0.456, 0.406]
        self.STD_PIX = [0.229, 0.224, 0.225]

        # create paths to dataset data folder and text files
        Datafolder = os.path.join(Dataset_dir,'Data')
        ImageNet_trainSet = os.path.join(Dataset_dir,'train.txt') # ~1.2M samples
        ImageNet_validSet = os.path.join(Dataset_dir,'val.txt')   # 50K samples
        ImageNet_testSet = os.path.join(Dataset_dir,'test.txt') # 100K samples, doesn't have GT labels! 
        class_map_txt = os.path.join(Dataset_dir,'synset_words.txt') # (synsetID, words) 
 
        # get useful mappings: integer->synsetID and synsetID->words 
        self.MAP_synsetID_words = {}
        self.MAP_int_synsetID = {}
        f_in = open(class_map_txt, 'r')    
        lines = f_in.readlines()
        class_idx = 0
        for line in lines:
            synsetID = line.split(' ')[0]
            words = line.replace(synsetID+' ', '').replace('\n','')       
            self.MAP_synsetID_words[synsetID] = words
            self.MAP_int_synsetID[class_idx] = synsetID
            class_idx = class_idx + 1
            
        # chose dataset txt file based on chosen dataset type
        self.DatasetType = DatasetType
        if self.DatasetType == 'train':
            f_in = open(ImageNet_trainSet,'r')
            
        elif self.DatasetType == 'val':
            f_in = open(ImageNet_validSet,'r')
            
        elif self.DatasetType == 'test':
            f_in = open(ImageNet_testSet,'r')
            
        # parse the chosen dataset txt file and get (image path -> GT label) 
        lines = f_in.readlines()
        for line in lines:
            fileName = line.split(' ')[0]
            GT_class = int(line.split(' ')[1]) # integer encoded GT class label
            img_path = os.path.join(Datafolder, DatasetType, fileName)
            # save sample
            self.dataset.append((img_path,GT_class))
        f_in.close()
        
        # define the image augmentaions for train and val/test samples 
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256, antialias=None),
                transforms.CenterCrop(244),
                transforms.Normalize(mean = self.MEAN_PIX, std = self.STD_PIX)
            ]
        )
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256, antialias=None),
                transforms.RandomCrop(244),
                transforms.Normalize(mean = self.MEAN_PIX, std = self.STD_PIX)
            ]
        )
    
    def __getitem__(self, idx):
        # extract properties of indexed sample
        img_path, GT_class = self.dataset[idx]
        
        # load image on path
        try:
            image = cv2.imread(img_path)
        except Exception as ex:
            print('Error opening file', img_path)
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            print(img_path)
            
        # BGR -> RGB by swapping the colour channels (cv2 loads it as BGR!)
        image = image[:, :, [2, 1, 0]]    
            
        # apply the image augmentations 
        if self.DatasetType == 'train':
            augImg = self.train_transform(image)
        elif self.DatasetType == 'val' or self.DatasetType == 'test':
            augImg = self.test_transform(image)
        
        # convert GT label to tensor (integer encoding)
        GT = torch.tensor([GT_class], dtype = torch.long)
        
        return augImg, GT , idx 
    
    def __len__(self):
        return len(self.dataset)
    

'''##########################################################################################################################
        Util function to convert the normalized torch tensor image to an image plotable with imshow()
##########################################################################################################################'''
def ConvertToPlotableImage(img, MEAN_PIX, STD_PIX):
        img = img.numpy() # convert tensor to numpy array
        img = np.transpose(img,(1,2,0)) # (CxHxW) -> (HxWxC)
        img = (img*STD_PIX) + MEAN_PIX # undo normalization
        img = (img * 255).astype(np.uint8) # pixel value [0,1] -> [0, 255]
        return img