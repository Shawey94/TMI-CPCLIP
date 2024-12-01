#CREDIT TO https://blog.csdn.net/qq_41234663/article/details/131024876

import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import random
import re
import cv2
import clip

#CLIP will process different sizes of image first, so may not need to do data augmentation for images
'''
self.transform = A.Compose([ 
                            A.Resize(height = 224, width = 224, p =1.0), 
                            A.HorizontalFlip(p = 0.5), 
                            A.RandomBrightnessContrast(p=0.2), 
                            A.Flip(p=0.2), 
                            A.Normalize(mean=(0.5,0.5,0.5),std=(0.3,0.3,0.3),p=1.0),
                            ])
'''


class CustomedMedicalData_MedCLIP(Dataset):
    def __init__(self,data_root,preprocess):
        self.data_root = data_root
        # 处理图像
        self.process = preprocess
        # 获得数据(根据自己的情况更改)
        self.samples = []
        self.sam_labels = []
        self.pure_labels = []

		# 5.2 获得所有的样本(根据自己的情况更改)
        with open(self.data_root,'r') as f:
            for line in f:
                img_path = line.split('^')[0]
                label = (line.split('^')[1]).strip()
                self.pure_labels.append(label)
                if('MIMICGAZE' in data_root):
                    label = label
                else:
                    label = "a figure of " + label
                self.samples.append(img_path)
                self.sam_labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        labels = self.pure_labels[idx]
        img_path = self.samples[idx]
        # 加载图像
        #image = Image.fromarray(cv2.imread(img_path))
        '''
        image = Image.open(img_path)
        inputs = self.process(
            text=labels, 
            images=image, 
            return_tensors="pt", 
            padding=True
            )
        '''
        return img_path, labels


class CustomedMedicalData(Dataset):
    def __init__(self,data_root,preprocess):
        self.data_root = data_root
        # 处理图像
        self.img_process = preprocess
        # 获得数据(根据自己的情况更改)
        self.samples = []
        self.sam_labels = []
        self.pure_labels = []
        self.heat_maps = []

		# 5.2 获得所有的样本(根据自己的情况更改)
        with open(self.data_root,'r') as f:
            for line in f:
                img_path = line.split('^')[0]
                label = (line.split('^')[1]).strip()
                if('MIMICGAZE' in self.data_root):
                    heatmap_path = (line.split('^')[2]).strip()
                    self.heat_maps.append(heatmap_path)
                self.pure_labels.append(label)
                if('MIMICGAZE' in data_root):
                    label = label
                else:
                    label = "a figure of " + label
                self.samples.append(img_path)
                self.sam_labels.append(label)
        # 转换为token
        self.tokens = clip.tokenize(self.sam_labels, context_length = 512, truncate=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        labels = self.pure_labels[idx]
        img_path = self.samples[idx]
        token = self.tokens[idx]
        # 加载图像
        image = Image.fromarray(cv2.imread(img_path))
        # 对图像进行转换
        image = self.img_process(image)   #

        if(self.heat_maps):
            heatmap_path = self.heat_maps[idx]
            heatmap = Image.fromarray(cv2.imread(heatmap_path))
            # 对图像进行转换
            heatmap = self.img_process(heatmap)   #
            return image, token, labels, heatmap

        return image, token, labels

class CustomedData(Dataset):
    def __init__(self,img_root,dataset,preprocess):
        # 1.根目录(根据自己的情况更改)
        self.img_root = img_root
        # 2.训练图片和测试图片地址(根据自己的情况更改)
        self.datasetset_file = os.path.join( './data/' +dataset+'.txt')
        # 4.处理图像
        self.img_process = preprocess
        # 5.获得数据(根据自己的情况更改)
        self.samples = []
        self.sam_labels = []
        self.pure_labels = []
        # 5.1 训练还是测试数据集
        self.read_file = ""
		# 5.2 获得所有的样本(根据自己的情况更改)
        with open(self.datasetset_file,'r') as f:
            for line in f:
                img_path = line.split('^')[0]
                label = (line.split('^')[1]).strip()
                self.pure_labels.append(label)
                label = "a photo of " + label
                self.samples.append(img_path)
                self.sam_labels.append(label)
        # 转换为token
        self.tokens = clip.tokenize(self.sam_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        labels = self.pure_labels[idx]
        img_path = self.samples[idx]
        token = self.tokens[idx]
        # 加载图像
        if('jpg' not in img_path):
            image = Image.fromarray(np.load(img_path))
        else:
            image = Image.fromarray(cv2.imread(img_path))
        # 对图像进行转换
        image = self.img_process(image)   #
        return image, token, labels

class CustomedImageNet(Dataset):
    def __init__(self,img_root,is_train,preprocess):
        # 1.根目录(根据自己的情况更改)
        self.img_root = img_root
        # 2.训练图片和测试图片地址(根据自己的情况更改)
        self.train_set_file = os.path.join(img_root,'train.txt')
        self.test_set_file = os.path.join(img_root,'test.txt')
        # 3.训练 or 测试(根据自己的情况更改)
        self.is_train = is_train
        # 4.处理图像
        self.img_process = preprocess
        # 5.获得数据(根据自己的情况更改)
        self.samples = []
        self.sam_labels = []
        self.pure_labels = []
        # 5.1 训练还是测试数据集
        self.read_file = ""
        if is_train:
            self.read_file = self.train_set_file
        else:
            self.read_file = self.test_set_file
		# 5.2 获得所有的样本(根据自己的情况更改)
        with open(self.read_file,'r') as f:
            for line in f:
                img_path = line.split(' ')[0]
                label = line.split('.')[1].split('/')[-1]
                label = re.sub(r'[0-9]+', '', label)
                label = label.replace('_', ' ')
                label = label.strip(' ')
                self.pure_labels.append(label)
                label = "a photo of " + label
                self.samples.append(img_path)
                self.sam_labels.append(label)
        # 转换为token
        self.tokens = clip.tokenize(self.sam_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        labels = self.pure_labels[idx]
        img_path = self.samples[idx]
        token = self.tokens[idx]
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        # 对图像进行转换
        image = self.img_process(image)
        return image, token, labels


if __name__ == "__main__":

    model, preprocess = clip.load("ViT-B/32")

    dataset_name = 'CIFAR10'
    dataset = CustomedData(img_root= './data/'+dataset_name, is_train = False, preprocess = preprocess)
    data_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 4,shuffle= True)
    for images,label_tokens,labels in data_loader:
        class_embeddings = model.encode_text(label_tokens[0].cuda())
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()
        print('images.shape:', images.shape)
        break

    dataset_name = 'ImageNet'
    dataset = CustomedImageNet(img_root= './data/'+dataset_name, is_train = False, preprocess = preprocess)
    data_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 4,shuffle= True)
    for images,label_tokens,labels in data_loader:
        print('images.shape:', images.shape)
        break


    
