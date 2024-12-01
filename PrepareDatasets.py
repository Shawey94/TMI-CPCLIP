# RUN this file to download and prepare files, then you can run dataloader!
# Attention, after read JPEG and save JPEG, image file size will shrink, this is because JPEG is lossy image 
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import cv2 as cv
import numpy as np
import torch
from parameters import *
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from utils import *
import matplotlib.pyplot as plt 
from PIL import Image
import shutil
from PIL import Image
import clip
import json
import csv
import pandas as pd

def delete_lines(filename, head,tail): #remove the first several and last rows
    fin = open(filename, 'r')
    a = fin.readlines()
    fout = open(filename, 'w')
    b = ''.join(a[head:-tail])
    fout.write(b)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (img, label ,path)

#ImageNet##########################################################################################
def get_imagenet(root, target_transform = None):
        transform_train = transforms.Compose([
            #transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_val = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tra_root = os.path.join(root,'train')
        trainset = datasets.ImageFolder(root=tra_root,
                                transform=transform_train,
                                target_transform=target_transform)
        val_root = os.path.join(root,'val')
        valset = datasets.ImageFolder(root=val_root,
                                transform=transform_val,
                                target_transform=target_transform)
        return trainset,valset

def get_loader(root):
    trainset, testset = get_imagenet(root)

    train_sampler = RandomSampler(trainset) 
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=1,
                              num_workers=1,
                              #shuffle =True,
                              )
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=1,
                             num_workers=1,
                             #shuffle =True,
                             ) if testset is not None else None
    return train_loader, test_loader

def PrepareImageNet():
    name = 'ImageNet'
    classes2labels_txtpath = './data/'+name+'/ImageNetClassesLables.txt'
    if(not os.path.exists('./data/'+name+'/Train')):
        os.makedirs('./data/'+name+'/Train')
    if(not os.path.exists('./data/'+name+'/Test')):
        os.makedirs('./data/'+name+'/Test')
    
    f = open(classes2labels_txtpath, "r")
    class_lookup_table = {}
    for x in f:
        classname = x.split(' ')[0]
        label = x.split(' ')[2][0:-1]
        label = label.replace('_', ' ')
        if classname not in class_lookup_table.keys():
            class_lookup_table[classname] = label

    #train_sampler = RandomSampler(trainset) 
    #test_sampler = SequentialSampler(testset)

    class_count = {}
    filename = './data/'+name+'/test.txt'
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('test txt created!')

    transform = transforms.Compose([transforms.ToTensor()]) # my transformations.
    dataset = ImageFolderWithPaths(root = os.path.join(opt.imagenet_path,'val'),transform=transform) # add transformation directly
    test_loader = DataLoader(dataset,
                             #sampler=test_sampler,
                             #batch_size=1,
                             #num_workers=1,
                             #shuffle =True,
                             )
    for te_transformed_normalized_img, te_labels, paths in test_loader:
        print(te_transformed_normalized_img.shape, te_labels, paths)
        label = imagenet_classes[te_labels]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
        te_transformed_normalized_img = np.array(te_transformed_normalized_img).squeeze()
        img = np.transpose(te_transformed_normalized_img,(1,2,0))
        #plt.imshow( np.transpose( img, (0,1,2)))
        #plt.show()
        temp_label = label.replace(' ','_')
        shutil.copyfile(paths[0], './data/ImageNet/Test/'+temp_label+'_'+str(class_count[label])+'.JPEG')
        with open(filename,'a+') as f:    #设置文件对象
            f.write('./data/'+name+'/Test/'+temp_label+'_'+str(class_count[label])+'.JPEG'+' '+label)                 #将字符串写入文件中
            f.write('\n')# 换行
    print(class_count)
    count = len(open(filename,'rU').readlines())
    print('test.txt lines: ', count)


    class_count = {}
    filename = './data/'+name+'/train.txt'
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('train txt created!')
                    
    dataset = ImageFolderWithPaths(root = os.path.join(opt.imagenet_path,'train'),transform=transform) # add transformation directly
    train_loader = DataLoader(dataset)
    for tra_transformed_normalized_img, tra_labels, paths in train_loader:
        print(tra_transformed_normalized_img.shape, tra_labels, paths)
        label = imagenet_classes[tra_labels]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
        tra_transformed_normalized_img = np.array(tra_transformed_normalized_img).squeeze()
        img = np.transpose(tra_transformed_normalized_img,(1,2,0))
        #plt.imshow( np.transpose( img, (0,1,2)))
        temp_label = label.replace(' ','_')
        shutil.copyfile(paths[0], './data/ImageNet/Train/'+temp_label+'_'+str(class_count[label])+'.JPEG')
        with open(filename,'a+') as f:    #设置文件对象
            f.write('./data/'+name+'/Train/'+temp_label+'_'+str(class_count[label])+'.JPEG'+' '+label)                 #将字符串写入文件中
            f.write('\n')# 换行
    print(class_count)
    count = len(open(filename,'rU').readlines())
    print('train.txt lines: ', count)


    '''
    train_loader, test_loader = get_loader(opt.imagenet_path)

    class_count = {}
    filename = './data/'+name+'/test.txt'
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('test txt created!')
            
    for i, (te_transformed_normalized_img, te_labels) in enumerate(test_loader):
        label = imagenet_classes[te_labels]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
        te_transformed_normalized_img = np.array(te_transformed_normalized_img).squeeze()
        img = np.transpose(te_transformed_normalized_img,(1,2,0))
        plt.imshow( np.transpose( img, (0,1,2)))
        #plt.show()
        img = img * 255
        img = img.astype(np.uint8)
        im = Image.fromarray(img)
        im.save('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.JPEG') #
        with open(filename,'a+') as f:    #设置文件对象
            f.write('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.JPEG'+' '+label)                 #将字符串写入文件中
            f.write('\n')# 换行
    print(class_count)
    count = len(open(filename,'rU').readlines())
    print('test.txt lines: ', count)

    class_count = {}
    filename = './data/'+name+'/train.txt'
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('train txt created!')
    
    for i, (tra_transformed_normalized_img, tra_labels) in enumerate(train_loader):
        label = imagenet_classes[tra_labels]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
        tra_transformed_normalized_img = np.array(tra_transformed_normalized_img).squeeze()
        img = np.transpose(tra_transformed_normalized_img,(1,2,0))
        plt.imshow( np.transpose( img, (0,1,2)))
        #plt.show()
        #通道顺序为RGB
        #img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        img = img * 255
        img = img.astype(np.uint8)
        im = Image.fromarray(img)
        im.save('./data/'+name+'/Train/'+label+'_'+str(class_count[label])+'.JPEG')
        #cv.imwrite('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.JPEG', img)
        with open(filename,'a+') as f:    #设置文件对象
            f.write('./data/'+name+'/Train/'+label+'_'+str(class_count[label])+'.JPEG'+' '+label)                 #将字符串写入文件中
            f.write('\n')# 换行
    print(class_count)
    count = len(open(filename,'rU').readlines())
    print('train.txt lines: ', count)
    '''

####################################################################################################

#CIFAR 10###########################################################################################
def PrepareCIFAR10V2():
    name = 'CIFAR10'
    if(not os.path.exists('./data/'+name)):
        os.makedirs('./data/'+name+'/Train')
        os.makedirs('./data/'+name+'/Test')
    #CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    train_loader = torch.utils.data.DataLoader(dataset = cifar_trainset,batch_size = 1,shuffle= True)
    test_loader = torch.utils.data.DataLoader(dataset = cifar_testset,batch_size = 1,shuffle= True)
    classes = cifar_testset.classes
    print(cifar_testset.classes)

    class_count = {}
    filename = './data/'+'/cifar10.txt'
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('cifar10 txt created!')
    for img, index in cifar_testset:
        label = classes[index]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
        #model, preprocess = clip.load('RN50', device=device,jit=False)
        #img = preprocess(img)
        np.save('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.npy', img)
        #img = np.load('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.npy')
        with open(filename,'a+') as f:    #设置文件对象
            f.write('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.npy'+'^'+label)                 #将字符串写入文件中
            f.write('\n')# 换行

    for img, index in cifar_trainset:
        label = classes[index]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
        #model, preprocess = clip.load('RN50', device=device,jit=False)
        #img = preprocess(img)
        np.save('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.npy', img)
        #img = np.load('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.npy')
        with open(filename,'a+') as f:    #设置文件对象
            f.write('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.npy'+'^'+label)                 #将字符串写入文件中
            f.write('\n')# 换行

    print(class_count)
    count = len(open(filename,'rU').readlines())
    print('cifar10.txt lines: ', count)

####################################################################################################




#CIFAR 100###########################################################################################
def PrepareCIFAR100V2():
    name = 'CIFAR100'
    if(not os.path.exists('./data/'+name)):
        os.makedirs('./data/'+name+'/Train')
        os.makedirs('./data/'+name+'/Test')

    cifar_testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=None)
    cifar_trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=None)
    train_loader = torch.utils.data.DataLoader(dataset = cifar_trainset,batch_size = 1,shuffle= True)
    test_loader = torch.utils.data.DataLoader(dataset = cifar_testset,batch_size = 1,shuffle= True)
    classes = cifar_testset.classes
    print(cifar_testset.classes)

    class_count = {}
    filename = './data/'+'/cifar100.txt'
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('cifar100 txt created!')
    for img, index in test_loader.dataset: #img 32* 32
        label = classes[index]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
        np.save('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.npy', img)
        with open(filename,'a+') as f:    #设置文件对象
            f.write('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.npy'+'^'+label)                 #将字符串写入文件中
            f.write('\n')# 换行

    for img, index in train_loader.dataset:
        label = classes[index]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
        np.save('./data/'+name+'/Train/'+label+'_'+str(class_count[label])+'.npy', img)
        with open(filename,'a+') as f:    #设置文件对象
            f.write('./data/'+name+'/Train/'+label+'_'+str(class_count[label])+'.npy'+'^'+label)                 #将字符串写入文件中
            f.write('\n')# 换行

    print(class_count)
    count = len(open(filename,'rU').readlines())
    print('cifar100.txt lines: ', count)
############################################################################################

def PrepareMIMIC_GAZE():
    subs_file = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/MIMIC_GAZE/common_subjects.txt'

    f = open(subs_file, "r")
    lines = f.readlines()
    subs = []
    for line in lines:
        sub = line.strip()
        subs.append(sub)

    imgs_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/MIMIC_GAZE/images'
    texts_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/MIMIC_GAZE/transcripts/all_transcripts_jsons'
    heat_maps_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/MIMIC_GAZE/heatmaps_new'

    imgs = os.listdir(imgs_path)
    texts = os.listdir(texts_path)
    heat_maps = os.listdir(heat_maps_path)
    common_subs = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/MIMIC_GAZE/common_subjects.txt'

    temp_f = open(common_subs, "r")
    temp = temp_f.readlines()
    common_subs = []
    for sub in temp:
        common_subs.append(sub.strip())
    temp_f.close()

    filename = './data/MIMICGAZE_CLIP_train.txt'    # img path, and corresponding texts
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('MIMICGAZE_CLIP_train txt created!')

    for img in imgs:
        with open(filename,'a+') as f:    #设置文件对象
            sub_id = (img.strip()).split('.')[0]
            if(sub_id in common_subs):
                heat_map = 'static_heatmap.jpg'

                text = os.listdir(texts_path+'/'+sub_id)[0]
                # Opening JSON file
                temp_f = open(texts_path+'/'+sub_id+'/'+text)
                # returns JSON object as 
                # a dictionary
                data = json.load(temp_f)
                # Iterating through the json
                # list
                sub_texts = data['full_text']
                # Closing file
                temp_f.close()

                f.write(imgs_path+'/'+img.strip()+'^'+sub_texts[1:]+'^'+heat_maps_path+'/'+sub_id+'/'+heat_map) #use _ not space to seperate, because there are spaces in sentences
                f.write('\n')# 换行

    count = len(open(filename,'rU').readlines())
    print('train.txt lines: ', count)

def PrepareINbreast():

    imgs_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/INbreast'

    train_imgs_path = imgs_path + '/' + 'train'
    test_imgs_path = imgs_path + '/' + 'test'

    train_imgs = os.listdir(train_imgs_path)
    test_imgs = os.listdir(test_imgs_path)
    
    classes = INbreast_classes

    filename = './data/INbreast.txt'    # img path, and corresponding texts
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('INbreast txt created!')

    for img in train_imgs:
        with open(filename,'a+') as f:    #设置文件对象
            if(img.split('_')[2][0] == 'l'):
                label = img.split('_')[2][1]
            else: 
                label = img.split('_')[3][1]
            f.write(imgs_path+'/train/'+img.strip()+'^'+classes[int(label)]) #use ^ not space to seperate, because there are spaces in sentences
            f.write('\n')# 换行
    for img in test_imgs:
        with open(filename,'a+') as f:    #设置文件对象
            if(img.split('_')[2][0] == 'l'):
                label = img.split('_')[2][1]
            else: 
                label = img.split('_')[3][1]
            f.write(imgs_path+'/test/'+img.strip()+'^'+classes[int(label)]) #use ^ not space to seperate, because there are spaces in sentences
            f.write('\n')# 换行

    count = len(open(filename,'rU').readlines())
    print('INbreast.txt lines: ', count)

def PrepareSIIMACR():
    imgs_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/SIIM-ACR'

    train_imgs_path = imgs_path + '/' + 'train_list.csv'
    test_imgs_path = imgs_path + '/' + 'test_list.csv'

    classes = SIIMACR_classes

    filename = './data/SIIMACR.txt'    # img path, and corresponding texts
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('SIIMACR txt created!')

    with open(filename,'a+') as f:    #设置文件对象
        with open(train_imgs_path, mode ='r')as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                img = (lines[0]).split('/')[-1]
                label = lines[1]

                f.write(imgs_path+'/'+img+'^'+classes[int(label)]) #use _ not space to seperate, because there are spaces in sentences
                f.write('\n')# 换行
        with open(test_imgs_path, mode ='r')as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                img = (lines[0]).split('/')[-1]
                label = lines[1]

                f.write(imgs_path+'/'+img+'^'+classes[int(label)]) #use _ not space to seperate, because there are spaces in sentences
                f.write('\n')# 换行
    count = len(open(filename,'rU').readlines())
    print('INbreast.txt lines: ', count)


def PrepareChexPert5x200():
    imgs_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/chexpert5x200/'
    csv_path = imgs_path + 'chexpert_5x200.csv'

    classes = cheXpert_classes

    filename = './data/chexpert5x200.txt'    # img path, and corresponding texts
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('chexpert5x200 txt created!')
   
    with open(filename,'a+') as f:    #设置文件对象
        with open(csv_path, mode ='r')as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                if('Path' in lines and 'Edema' in lines and 'Consolidation' in lines):
                    continue
                img = lines[0]
                if(lines[1] == '1'):
                    label = 0
                if(lines[2] == '1'):
                    label = 1
                if(lines[3] == '1'):
                    label = 2
                if(lines[4] == '1'):
                    label = 3
                if(lines[5] == '1'):
                    label = 4
                temp = imgs_path+(img.split('/'))[0] + '/' + (img.split('/'))[1] +  '/' + (img.split('/'))[2]
                if(not os.path.exists(temp)):
                    continue
                f.write(imgs_path+img+'^'+classes[int(label)]) #use _ not space to seperate, because there are spaces in sentences
                f.write('\n')# 换行

    count = len(open(filename,'rU').readlines())
    print('CheXpert5x200.txt lines: ', count)

def PrepareChestXray():
    imgs_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/chest_xray/'
     
    tra_path = imgs_path + 'train'
    val_path = imgs_path + 'val'
    test_path = imgs_path + 'test'

    tra_normal_imgs = os.listdir(tra_path+'/NORMAL')
    tra_pneumonia_imgs = os.listdir(tra_path+'/PNEUMONIA')
    val_normal_imgs = os.listdir(val_path+'/NORMAL')
    val_pneumonia_imgs = os.listdir(val_path+'/PNEUMONIA')
    te_normal_imgs = os.listdir(test_path+'/NORMAL')
    te_pneumonia_imgs = os.listdir(test_path+'/PNEUMONIA')

    classes = chest_xray_classes

    filename = './data/chestxray.txt'    # img path, and corresponding texts
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('chestxray txt created!')
   
    with open(filename,'a+') as f:    #设置文件对象
        for img in tra_normal_imgs:
            temp = tra_path+'/NORMAL/'+img 
            f.write(temp+'^'+'normal') #use _ not space to seperate, because there are spaces in sentences
            f.write('\n')# 换行
        for img in tra_pneumonia_imgs:
            temp = tra_path+'/PNEUMONIA/'+img 
            f.write(temp+'^'+'pneumonia') #use _ not space to seperate, because there are spaces in sentences
            f.write('\n')# 换行
        for img in val_normal_imgs:
            temp = val_path+'/NORMAL/'+img 
            f.write(temp+'^'+'normal') #use _ not space to seperate, because there are spaces in sentences
            f.write('\n')# 换行
        for img in val_pneumonia_imgs:
            temp = val_path+'/PNEUMONIA/'+img 
            f.write(temp+'^'+'pneumonia') #use _ not space to seperate, because there are spaces in sentences
            f.write('\n')# 换行
        for img in te_normal_imgs:
            temp = test_path+'/NORMAL/'+img 
            f.write(temp+'^'+'normal') #use _ not space to seperate, because there are spaces in sentences
            f.write('\n')# 换行
        for img in te_pneumonia_imgs:
            temp = test_path+'/PNEUMONIA/'+img 
            f.write(temp+'^'+'pneumonia') #use _ not space to seperate, because there are spaces in sentences
            f.write('\n')# 换行


    count = len(open(filename,'rU').readlines())
    print('chestxray.txt lines: ', count)



def PrepareFlickr8k():
    flick8k_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/Flickr8k/'
    img_path = flick8k_path + 'Flicker8k_Dataset'
    text_file = flick8k_path + 'Flickr8k.token.txt'
    
    flick_imgs = os.listdir(img_path)

    filename = './data/flickr8k.txt'    # img path, and corresponding texts
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('flickr8k txt created!')

    
    f2 = open(text_file,"r")
    texts = f2.readlines()

    with open(filename,'a+') as f:    #设置文件对象
        for img in flick_imgs:
            indices = [index for index, item in enumerate(texts) if img in item]
            tmp_text = (texts[indices[0]].split('\t')[1]).strip()
            temp = img_path+'/'+img 
            f.write(temp+'^'+tmp_text) #use _ not space to seperate, because there are spaces in sentences
            f.write('\n')# 换行


    count = len(open(filename,'rU').readlines())
    print('flickr.txt lines: ', count)

def PrepareSTL10():
    name = 'STL10'
    if(not os.path.exists('./data/'+name)):
        os.makedirs('./data/'+name+'/Train')
        os.makedirs('./data/'+name+'/Test')
    
    stl10_testset = datasets.STL10(root='./data', split='train', download=True, transform=None)
    stl10_trainset = datasets.STL10(root='./data', split='test', download=True, transform=None)
    train_loader = torch.utils.data.DataLoader(dataset = stl10_trainset,batch_size = 1,shuffle= True)
    test_loader = torch.utils.data.DataLoader(dataset = stl10_testset,batch_size = 1,shuffle= True)
    classes = stl10_testset.classes
    print(stl10_testset.classes)

    class_count = {}
    filename = './data/'+'/stl10.txt'
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('test txt created!')
    for img, index in stl10_testset:
        label = classes[index]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
        #model, preprocess = clip.load('RN50', device=device,jit=False)
        #img = preprocess(img)
        np.save('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.npy', img)
        #img = np.load('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.npy')
        with open(filename,'a+') as f:    #设置文件对象
            f.write('./data/'+name+'/Test/'+label+'_'+str(class_count[label])+'.npy'+'^'+label)                 #将字符串写入文件中
            f.write('\n')# 换行
    
    for img, index in stl10_trainset:
        label = classes[index]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
        #img = np.transpose(img,(0,1,2))
        np.save('./data/'+name+'/Train/'+label+'_'+str(class_count[label])+'.npy', img)
        with open(filename,'a+') as f:    #设置文件对象
            f.write('./data/'+name+'/Train/'+label+'_'+str(class_count[label])+'.npy'+'^'+label)                 #将字符串写入文件中
            f.write('\n')# 换行

    print(class_count)
    count = len(open(filename,'rU').readlines())
    print('stl10.txt lines: ', count)

def PrepareTMED2():

    imgs_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/TMEDlabeled'
    csv_path = '/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/MICCAI24/WorkWithHuang/DEV479/TMED2_fold0_labeledpart.csv'
    dataset = pd.read_csv(csv_path)
    #p_id = dataset['query_key'].str.split("_", n = 1, expand = True)
    p_id = list(dataset['query_key'])
    p_view = list(dataset['view_label'])
    p_label = list(dataset['diagnosis_label'])

    relavent_p_id = []
    relavent_label = []
    for i in range(len(p_view)):
        if(p_view[i] in ['PLAX', 'PSAX']):
            relavent_p_id.append(p_id[i])
            if(p_label[i] == 'no_AS' or p_label[i] == 'severe_AS'):
                relavent_label.append(p_label[i])
            else:
                relavent_label.append('mild_AS')

    TMED2_imgs = os.listdir(imgs_path)
    
    classes = TMED2_classes

    filename = './data/TMED2.txt'    # img path, and corresponding texts
    if os.path.exists(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        with open(filename, mode='w', encoding='utf-8'):
            print('TMED2 txt created!')

    for img in TMED2_imgs:
        with open(filename,'a+') as f:      
            if(img in relavent_p_id):  
                pos = relavent_p_id.index(img)    
                label = relavent_label[pos]  
                print(label)                                             #设置文件对象
                f.write(imgs_path+'/'+img.strip()+'^'+label) #use ^ not space to seperate, because there are spaces in sentences
                f.write('\n')# 换行

    count = len(open(filename,'rU').readlines())
    print('TMED2.txt lines: ', count)

if __name__ == "__main__":

    # PrepareCIFAR10V2()    #convert from npy to jpg could impact performance 
    # PrepareCIFAR100V2()
    # PrepareSTL10()

    '''
    root_path = opt.imagenet_path
    train_loader, test_loader = get_loader(root_path)
    for i, (tra_transformed_normalized_img, tra_labels) in enumerate(train_loader):
        print(i)
        print(tra_labels)
        plt.imshow( np.transpose( tra_transformed_normalized_img[0], (1,2,0)))
        plt.show()
    '''

    #PrepareMIMIC_GAZE()   #/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/CP_CLIP/data/MIMIC_GAZE
    
    #PrepareINbreast()   

    #PrepareSIIMACR()

    #PrepareChexPert5x200()

    #PrepareChestXray()

    #PrepareFlickr8k()

    PrepareTMED2()