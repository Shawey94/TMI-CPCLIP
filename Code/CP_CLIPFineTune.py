#CREDIT TO https://blog.csdn.net/qq_41234663/article/details/131024876

# when FrozenCLIP, only finetune cp nn, the learning rate should be larger, such as 1e-3
# use cifar and imagenet label templates
# 10/11 make codes more efficient

#12/10/2023 use medical datasets
#findings: 12/20/2023 # normalized features before cp network is good for CheXpert
#          12/20/2023 # normalized features after cp network is good for Inbreast and SIIMACR

import numpy as np
import torch
from pkg_resources import packaging
import clip
import os
import skimage
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from loguru import logger
import torchvision.datasets as datasets
from dataloader import *
from parameters import *
from utils import *

from collections import OrderedDict
import torch
from CP_NN import CP_NN

print(clip.available_models())

##############################################################################################

#code for testing the dimension of the output features of CLIP
# Download the dataset
'''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
cifar100 = datasets.CIFAR100(root=os.path.expanduser("./data"), download=True, train=False)

model, preprocess = clip.load("ViT-B/32", device=device)
# Prepare the inputs
image, class_id = cifar100[3637]
image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)   # 1* 512 
    text_features = model.encode_text(text_inputs)     # 100 * 512, 100 is because of 100 classes
'''
###############################################################################################

def FineTuneCPCLIPv2(dataset1, dataset2, dataset3, dataset4):
    model_name = "CP_CLIPFinetunedv2On_"+dataset1
    #different from FineTuneCLIPv2 that take information from cores and peripheries nodes; this version only take information from core nodes
    #torch.autograd.set_detect_anomaly(True)

    model, preprocess = clip.load(opt.vision_encoder, device=device,jit=False)
    #input_resolution = model.visual.input_resolution
    #context_length = model.context_length
    #vocab_size = model.vocab_size
    #logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
     
    
    for name, param in model.named_parameters():
        if('visual' not in name):
            param.requires_grad = False
        #print(f"{name}: {param.requires_grad}")
    

    #add Core-periphery NN after CLIP
    if('RN' in opt.vision_encoder):
        nodes = nodes_lookup_table['RN']
        cores = round(nodes * opt.ratio)
    elif('ViT' in opt.vision_encoder):
        nodes = nodes_lookup_table['ViT']
        cores = round(nodes * opt.ratio)
    cp_mask = get_cp_mask(nodes, cores)
    cp_network = CP_NN(in_dim=nodes, out_dim=nodes, CP_mask= torch.from_numpy(cp_mask).to(device), device=device)
    cp_network.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('clip params ', params/1e6)

    '''
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    '''

    #print(preprocess)
    #print(model)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr':opt.lr_clip},
                           {'params': cp_network.parameters(), 'lr': opt.lr_cpNN}], 
                           betas=(0.9,0.99),eps=1e-6,weight_decay=0.001)

    #scheduler = lr_scheduler.StepLR(
    #       optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-10, last_epoch=- 1, verbose=False)

    SIIMACR_stored_acc = 0
    INbreast_stored_acc = 0
    CheXpert_stored_acc = 0
    #load saved model
    saved_models = os.listdir('./models_saved')
        
    print('in FineTuneCPCLIPv2')

    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    if(dataset1 == 'CIFAR10'):
        cifar10_classes = get_cifar10_classes()
    elif(dataset1 == 'CIFAR100'):
        cifar100_classes = get_cifar100_classes()
    elif(dataset1 == 'ImageNet'):
        ImageNet_classes = imagenet_classes
    if(dataset2 == 'INbreast' or dataset3 == 'INbreast'):
        INbreast_classes = iNbreast_classes
    if(dataset2 == 'SIIMACR' or dataset3 == 'SIIMACR'):
        SIIMACR_classes = sIIMACR_classes

    if('CIFAR' in dataset1):
        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    elif('ImageNet' in dataset1):
        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)

    if('INbreast' in dataset2 or 'INbreast' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.INbreast_path, preprocess = preprocess)
        INbreast_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(INbreast_loader)
        print('INbreast dataset_size ', dataset_size)
    if('SIIMACR' in dataset2 or 'SIIMACR' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.SIIMACR_path, preprocess = preprocess)
        SIIMACR_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(SIIMACR_loader)
        print('SIIMACR dataset_size ', dataset_size)
    if('MIMICGAZE' in dataset1):
        dataset = CustomedMedicalData(data_root= opt.MIMICGAZE_path, preprocess = preprocess)
        MIMICGAZE_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        tra_loader = MIMICGAZE_loader
        dataset_size = len(MIMICGAZE_loader)
        print('MIMICGAZE dataset_size ', dataset_size)
    if('CheXpert5x200' in dataset4):
        dataset = CustomedMedicalData(data_root= opt.CheXpert5x200_path, preprocess = preprocess)
        CheXpert5x200_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(CheXpert5x200_loader)
        print('CheXpert5x200 dataset_size ', dataset_size)


    #Finetune
    phase = "train"

    for epoch in range( opt.epochs):
        total_loss = 0
        batch_num = 0
        model.train()
        cp_network.train()
        for images,label_tokens,labels, heatmaps in tra_loader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            with torch.set_grad_enabled(phase == "train"):
                #logits_per_image, logits_per_text = model(images, label_tokens)
                image_features = model.encode_image(images)
                text_features = model.encode_text(label_tokens)

                # normalized features
                #image_features = image_features / image_features.norm(dim=1, keepdim=True)
                #text_features = text_features / text_features.norm(dim=1, keepdim=True)

                ###################################
                #CP Network acts on imges features and text features
                image_features = cp_network(image_features)
                text_features = cp_network(text_features)
                ###################################

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                #take the features in cores
                image_features = image_features[:,0:cores]
                text_features = text_features[:,0:cores]

                # cosine similarity as logits
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                total_loss = total_loss + cur_loss
                if phase == "train":
                    cur_loss.backward()  #retain_graph=True
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        #clip.model.convert_weights(model)  #basically convert the CLIP model weight into float16.

            if batch_num % 5 == 0:
                logger.info('{} epoch:{} loss_clip:{}'.format(phase,epoch,cur_loss))
                #break   # !!!!!!!!debug purpose, pls comment this after debug.
        epoch_clip_loss = total_loss / batch_num
            #torch.save(model.state_dict(),f"{model_name}_epoch_{epoch}.pth")
            #logger.info(f"weights_{epoch} saved")
            #logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

        scheduler.step()
 
        model.eval()    
        cp_network.eval()
        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in SIIMACR_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in SIIMACR_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(SIIMACR_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((SIIMACR_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('SIIMACR acc: ', correct/total)
            acc = round(correct/total, 4)
            if(acc> SIIMACR_stored_acc):
                SIIMACR_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' in temp_file and 'finetune' in temp_file and 'SIIMACR'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and str(nodes)==temp_file.split('_')[-3] and 'FrozenCLIP' not in temp_file and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(SIIMACR_stored_acc)}_{model_name}_zeroshoton_SIIMACR_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        correct = 0
        total = 0
        normal_correct = 0
        benign_correct = 0
        malignant_correct = 0
        normal_total = 0
        benign_total = 0
        malignant_total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in INbreast_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in INbreast_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(INbreast_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((INbreast_classes[indices[0]] == labels[img_cnt]))

                    if(labels[img_cnt] == 'normal'):    #normal', 'benign', 'malignant'
                        normal_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            normal_correct += 1
                    if(labels[img_cnt] == 'benign'):    #normal', 'benign', 'malignant'
                        benign_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            benign_correct += 1
                    if(labels[img_cnt] == 'malignant'):    #normal', 'benign', 'malignant'
                        malignant_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            malignant_correct += 1

                    img_cnt += 1
            print('INbreast acc: ', correct/total)
            temp_normal_acc = round(normal_correct / normal_total, 2)
            temp_benign_acc = round(benign_correct / benign_total, 2)
            temp_malignant_acc = round(malignant_correct / malignant_total, 2)
            acc = round(correct/total, 4)
            if(acc> INbreast_stored_acc):
                INbreast_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' in temp_file and 'finetune' in temp_file and 'INbreast'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and str(nodes)==temp_file.split('_')[-3] and 'FrozenCLIP' not in temp_file and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(temp_normal_acc)}_{str(temp_benign_acc)}_{str(temp_malignant_acc)}_{str(INbreast_stored_acc)}_{model_name}_zeroshoton_INbreast_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in cheXpert_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in CheXpert5x200_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(cheXpert_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((cheXpert_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('CheXpert acc: ', correct/total)
            acc = round(correct/total, 4)
            if(acc> CheXpert_stored_acc):
                CheXpert_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' in temp_file and 'finetune' in temp_file and 'CheXPert5x200'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and str(nodes)==temp_file.split('_')[-3] and 'FrozenCLIP' not in temp_file and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(CheXpert_stored_acc)}_{model_name}_zeroshoton_CheXPert5x200_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")



######################################################################################################################################
def FineTuneCPCLIPv2_FrozenCLIP(dataset1, dataset2, dataset3, dataset4):
    model_name = "FineTuneCPCLIPv2_FrozenCLIPOn_"+dataset1
    #different from FineTuneCLIPv2 that take information from cores and peripheries nodes; this version only take information from core nodes
    #torch.autograd.set_detect_anomaly(True)

    model, preprocess = clip.load(opt.vision_encoder, device=device,jit=False)
    #input_resolution = model.visual.input_resolution
    #context_length = model.context_length
    #vocab_size = model.vocab_size
    #logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    #add Core-periphery NN after CLIP
    if('RN' in opt.vision_encoder):
        nodes = nodes_lookup_table['RN']
        cores = round(nodes * opt.ratio)
    elif('ViT' in opt.vision_encoder):
        nodes = nodes_lookup_table['ViT']
        cores = round(nodes * opt.ratio)
    cp_mask = get_cp_mask(nodes, cores)
    cp_network = CP_NN(in_dim=nodes, out_dim=nodes, CP_mask= torch.from_numpy(cp_mask).to(device), device=device)
    cp_network.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('clip params ', params/1e6)

    '''
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    '''

    #print(preprocess)
    #print(model)
    optimizer = optim.Adam([
                           {'params': cp_network.parameters(), 'lr': opt.lr_cpNN_fzCLIP}], 
                           betas=(0.9,0.99),eps=1e-6,weight_decay=0.001)

    #scheduler = lr_scheduler.StepLR(
    #       optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-10, last_epoch=- 1, verbose=False)

    SIIMACR_stored_acc = 0
    INbreast_stored_acc = 0
    CheXpert_stored_acc = 0
    #load saved model
    saved_models = os.listdir('./models_saved')
    print('in FinetuneCPCLIPv2 frozen CLIP!')    

    for param in model.parameters(): #frozen CLIP
        param.requires_grad = False

    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    if(dataset1 == 'CIFAR10'):
        cifar10_classes = get_cifar10_classes()
    elif(dataset1 == 'CIFAR100'):
        cifar100_classes = get_cifar100_classes()
    elif(dataset1 == 'ImageNet'):
        ImageNet_classes = imagenet_classes
    if(dataset2 == 'INbreast' or dataset3 == 'INbreast'):
        INbreast_classes = iNbreast_classes
    if(dataset2 == 'SIIMACR' or dataset3 == 'SIIMACR'):
        SIIMACR_classes = sIIMACR_classes

    if('CIFAR' in dataset1):
        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    elif('ImageNet' in dataset1):
        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)

    if('INbreast' in dataset2 or 'INbreast' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.INbreast_path, preprocess = preprocess)
        INbreast_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(INbreast_loader)
        print('INbreast dataset_size ', dataset_size)
    if('SIIMACR' in dataset2 or 'SIIMACR' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.SIIMACR_path, preprocess = preprocess)
        SIIMACR_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(SIIMACR_loader)
        print('SIIMACR dataset_size ', dataset_size)
    if('MIMICGAZE' in dataset1):
        dataset = CustomedMedicalData(data_root= opt.MIMICGAZE_path, preprocess = preprocess)
        MIMICGAZE_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        tra_loader = MIMICGAZE_loader
        dataset_size = len(MIMICGAZE_loader)
        print('MIMICGAZE dataset_size ', dataset_size)
    if('CheXpert5x200' in dataset4):
        dataset = CustomedMedicalData(data_root= opt.CheXpert5x200_path, preprocess = preprocess)
        CheXpert5x200_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(CheXpert5x200_loader)
        print('CheXpert5x200 dataset_size ', dataset_size)

    #Finetune
    phase = "train"

    for epoch in range( opt.epochs ):
        total_loss = 0
        batch_num = 0
        #model.train()
        cp_network.train()
        for images,label_tokens,labels, heatmaps in tra_loader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            with torch.set_grad_enabled(phase == "train"):
                #logits_per_image, logits_per_text = model(images, label_tokens)
                image_features = model.encode_image(images)
                text_features = model.encode_text(label_tokens)

                # normalized features
                #image_features = image_features / image_features.norm(dim=1, keepdim=True)
                #text_features = text_features / text_features.norm(dim=1, keepdim=True)

                ###################################
                #CP Network acts on imges features and text features
                image_features = cp_network(image_features)
                text_features = cp_network(text_features)
                ###################################

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                
                #take the features in cores
                image_features = image_features[:,0:cores]
                text_features = text_features[:,0:cores]

                # cosine similarity as logits
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                total_loss = total_loss + cur_loss
                if phase == "train":
                    cur_loss.backward()  #retain_graph=True
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        #clip.model.convert_weights(model)  #basically convert the CLIP model weight into float16.

            if batch_num % 5 == 0:
                logger.info('{} epoch:{} loss_clip:{}'.format(phase,epoch,cur_loss))
                #break   # !!!!!!!!debug purpose, pls comment this after debug.
        epoch_clip_loss = total_loss / batch_num
            #torch.save(model.state_dict(),f"{model_name}_epoch_{epoch}.pth")
            #logger.info(f"weights_{epoch} saved")
            #logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

        scheduler.step()

        model.eval()    
        cp_network.eval()
        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in SIIMACR_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in SIIMACR_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(SIIMACR_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((SIIMACR_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('SIIMACR acc: ',correct/total)
            acc = round(correct/total, 4)
            if(acc> SIIMACR_stored_acc):
                SIIMACR_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' in temp_file and 'finetune' in temp_file and 'SIIMACR'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(SIIMACR_stored_acc)}_{model_name}_zeroshoton_SIIMACR_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        correct = 0
        total = 0
        normal_correct = 0
        benign_correct = 0
        malignant_correct = 0
        normal_total = 0
        benign_total = 0
        malignant_total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in INbreast_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in INbreast_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(INbreast_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((INbreast_classes[indices[0]] == labels[img_cnt]))

                    if(labels[img_cnt] == 'normal'):    #normal', 'benign', 'malignant'
                        normal_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            normal_correct += 1
                    if(labels[img_cnt] == 'benign'):    #normal', 'benign', 'malignant'
                        benign_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            benign_correct += 1
                    if(labels[img_cnt] == 'malignant'):    #normal', 'benign', 'malignant'
                        malignant_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            malignant_correct += 1

                    img_cnt += 1
            print('INbreast acc: ',correct/total)
            temp_normal_acc = round(normal_correct / normal_total, 2)
            temp_benign_acc = round(benign_correct / benign_total, 2)
            temp_malignant_acc = round(malignant_correct / malignant_total, 2)
            acc = round(correct/total, 4)
            if(acc> INbreast_stored_acc):
                INbreast_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' in temp_file and 'finetune' in temp_file and 'INbreast'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(temp_normal_acc)}_{str(temp_benign_acc)}_{str(temp_malignant_acc)}_{str(INbreast_stored_acc)}_{model_name}_zeroshoton_INbreast_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in cheXpert_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in CheXpert5x200_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(cheXpert_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((cheXpert_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('CheXpert acc: ',correct/total)
            acc = round(correct/total, 4)
            if(acc> CheXpert_stored_acc):
                CheXpert_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' in temp_file and 'finetune' in temp_file and 'CheXpert5x200'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(CheXpert_stored_acc)}_{model_name}_zeroshoton_CheXpert5x200_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")



#######################################################################################################################################
def FineTuneCPCLIP(dataset1, dataset2, dataset3, dataset4):
    model_name = "CP_CLIPFinetunedOn_"+dataset1
    #torch.autograd.set_detect_anomaly(True)

    model, preprocess = clip.load(opt.vision_encoder, device=device,jit=False)
    #input_resolution = model.visual.input_resolution
    #context_length = model.context_length
    #vocab_size = model.vocab_size
    #logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    
    for name, param in model.named_parameters():
        if('visual' not in name):
            param.requires_grad = False
        #print(f"{name}: {param.requires_grad}")
    

    #add Core-periphery NN after CLIP
    if('RN' in opt.vision_encoder):
        nodes = nodes_lookup_table['RN']
        cores = round(nodes * opt.ratio)
    elif('ViT' in opt.vision_encoder):
        nodes = nodes_lookup_table['ViT']
        cores = round(nodes * opt.ratio)
    cp_mask = get_cp_mask(nodes, cores)
    cp_network = CP_NN(in_dim=nodes, out_dim=nodes, CP_mask= torch.from_numpy(cp_mask).to(device), device=device)
    cp_network.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('clip params ', params/1e6)

    '''
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    '''

    #print(preprocess)
    #print(model)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr':opt.lr_clip},
                           {'params': cp_network.parameters(), 'lr': opt.lr_cpNN}], 
                           betas=(0.9,0.99),eps=1e-6,weight_decay=0.001)

    #scheduler = lr_scheduler.StepLR(
    #       optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-10, last_epoch=- 1, verbose=False)

    SIIMACR_stored_acc = 0
    INbreast_stored_acc = 0
    CheXpert5x200_stored_acc = 0
    #load saved model
    saved_models = os.listdir('./models_saved')
        
    print('in Finetune CP CLIP')

    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    if(dataset1 == 'CIFAR10'):
        cifar10_classes = get_cifar10_classes()
    elif(dataset1 == 'CIFAR100'):
        cifar100_classes = get_cifar100_classes()
    elif(dataset1 == 'ImageNet'):
        ImageNet_classes = imagenet_classes
    if(dataset2 == 'INbreast' or dataset3 == 'INbreast'):
        INbreast_classes = iNbreast_classes
    if(dataset2 == 'SIIMACR' or dataset3 == 'SIIMACR'):
        SIIMACR_classes = sIIMACR_classes

    if('CIFAR' in dataset1):
        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    elif('ImageNet' in dataset1):
        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)

    if('INbreast' in dataset2 or 'INbreast' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.INbreast_path, preprocess = preprocess)
        INbreast_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(INbreast_loader)
        print('INbreast dataset_size ', dataset_size)
    if('SIIMACR' in dataset2 or 'SIIMACR' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.SIIMACR_path, preprocess = preprocess)
        SIIMACR_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(SIIMACR_loader)
        print('SIIMACR dataset_size ', dataset_size)
    if('MIMICGAZE' in dataset1):
        dataset = CustomedMedicalData(data_root= opt.MIMICGAZE_path, preprocess = preprocess)
        MIMICGAZE_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        tra_loader = MIMICGAZE_loader
        dataset_size = len(MIMICGAZE_loader)
        print('MIMICGAZE dataset_size ', dataset_size)
    if('CheXpert5x200' in dataset4):
        dataset = CustomedMedicalData(data_root= opt.CheXpert5x200_path, preprocess = preprocess)
        CheXpert5x200_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(CheXpert5x200_loader)
        print('CheXpert5x200 dataset_size ', dataset_size)


    #Finetune
    phase = "train"

    for epoch in range( opt.epochs):
        total_loss = 0
        batch_num = 0
        model.train()
        cp_network.train()
        for images,label_tokens,labels, heatmaps in tra_loader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            with torch.set_grad_enabled(phase == "train"):
                #logits_per_image, logits_per_text = model(images, label_tokens)
                image_features = model.encode_image(images)
                text_features = model.encode_text(label_tokens)  #ViT dim is 512, ResNet dim is 1024

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                ###################################
                #CP Network acts on imges features and text features
                image_features = cp_network(image_features)
                text_features = cp_network(text_features)
                ###################################

                # normalized features
                #image_features = image_features / image_features.norm(dim=1, keepdim=True)
                #text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                total_loss = total_loss + cur_loss
                if phase == "train":
                    cur_loss.backward()  #retain_graph=True
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        #clip.model.convert_weights(model)

            if batch_num % 3 == 0:
                logger.info('{} epoch:{} loss_clip:{}'.format(phase,epoch,cur_loss))
                #break   # !!!!!!!!debug purpose, pls comment this after debug.
        epoch_clip_loss = total_loss / batch_num
            #torch.save(model.state_dict(),f"{model_name}_epoch_{epoch}.pth")
            #logger.info(f"weights_{epoch} saved")
            #logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
        
        scheduler.step()

        model.eval()    
        
        cp_network.eval()
        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in SIIMACR_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in SIIMACR_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(SIIMACR_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((SIIMACR_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('SIIMACR acc is ', correct/total)
            acc = round(correct/total, 4)
            if(acc> SIIMACR_stored_acc):
                SIIMACR_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' not in temp_file and 'finetune' in temp_file and 'SIIMACR'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and 'FrozenCLIP' not in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(SIIMACR_stored_acc)}_{model_name}_zeroshot_on_SIIMACR_{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        correct = 0
        total = 0
        normal_correct = 0
        benign_correct = 0
        malignant_correct = 0
        normal_total = 0
        benign_total = 0
        malignant_total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in INbreast_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in INbreast_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(INbreast_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((INbreast_classes[indices[0]] == labels[img_cnt]))

                    if(labels[img_cnt] == 'normal'):    #normal', 'benign', 'malignant'
                        normal_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            normal_correct += 1
                    if(labels[img_cnt] == 'benign'):    #normal', 'benign', 'malignant'
                        benign_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            benign_correct += 1
                    if(labels[img_cnt] == 'malignant'):    #normal', 'benign', 'malignant'
                        malignant_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            malignant_correct += 1

                    img_cnt += 1
            print('INbreast acc is ', correct/total)
            temp_normal_acc = round(normal_correct / normal_total, 2)
            temp_benign_acc = round(benign_correct / benign_total, 2)
            temp_malignant_acc = round(malignant_correct / malignant_total, 2)
            acc = round(correct/total, 4)
            if(acc> INbreast_stored_acc):
                INbreast_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' not in temp_file and 'finetune' in temp_file and 'INbreast'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and 'FrozenCLIP' not in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(temp_normal_acc)}_{str(temp_benign_acc)}_{str(temp_malignant_acc)}_{str(INbreast_stored_acc)}_{model_name}_zeroshot_on_INbreast_{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        
        correct = 0
        total = 0
        temp_cardi = 0
        temp_edema = 0
        temp_consol = 0 
        temp_atelect = 0
        temp_pleural = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in cheXpert_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in CheXpert5x200_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(cheXpert_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((cheXpert_classes[indices[0]] == labels[img_cnt]))
                    if(cheXpert_classes[indices[0]] == 'cardiomegaly'):
                        temp_cardi +=1
                    elif(cheXpert_classes[indices[0]] == 'edema'):
                        temp_edema += 1
                    elif(cheXpert_classes[indices[0]] == 'consolidation'):
                        temp_consol += 1
                    elif(cheXpert_classes[indices[0]] == 'atelectasis'):
                        temp_atelect += 1
                    elif(cheXpert_classes[indices[0]] == 'pleural effusion'):
                        temp_pleural += 1
                    img_cnt += 1
            print('cardimegaly {}, edema {}, consolidation {}, atelectasis {}, pleural effusion {}'.format(temp_cardi, temp_edema, temp_consol, temp_atelect, temp_pleural))
            print('CheXpert acc is ', correct/total)
            acc = round(correct/total, 4)
            if(acc> CheXpert5x200_stored_acc):
                CheXpert5x200_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' not in temp_file and 'finetune' in temp_file and 'CheXpert5x200'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and 'FrozenCLIP' not in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(CheXpert5x200_stored_acc)}_{model_name}_zeroshot_on_CheXpert5x200_{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")
        

#######################################################################################################################################
def FineTuneCPCLIP_FrozenCLIP(dataset1, dataset2, dataset3, dataset4):    
    model_name = "FineTuneCPCLIP_FrozenCLIPOn_"+dataset1
    #torch.autograd.set_detect_anomaly(True)

    model, preprocess = clip.load(opt.vision_encoder, device=device,jit=False)
    #input_resolution = model.visual.input_resolution
    #context_length = model.context_length
    #vocab_size = model.vocab_size
    #logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    #add Core-periphery NN after CLIP
    if('RN' in opt.vision_encoder):
        nodes = nodes_lookup_table['RN']
        cores = round(nodes * opt.ratio)
    elif('ViT' in opt.vision_encoder):
        nodes = nodes_lookup_table['ViT']
        cores = round(nodes * opt.ratio)
    cp_mask = get_cp_mask(nodes, cores)
    cp_network = CP_NN(in_dim=nodes, out_dim=nodes, CP_mask= torch.from_numpy(cp_mask).to(device), device=device)
    cp_network.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('clip params ', params/1e6)

    '''
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    '''

    #print(preprocess)
    #print(model)
    optimizer = optim.Adam([
                           {'params': cp_network.parameters(), 'lr': opt.lr_cpNN_fzCLIP}], 
                           betas=(0.9,0.99),eps=1e-6,weight_decay=0.001)

    #scheduler = lr_scheduler.StepLR(
    #       optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-10, last_epoch=- 1, verbose=False)

    SIIMACR_stored_acc = 0
    INbreast_stored_acc = 0
    CheXpert5x200_stored_acc = 0
    #load saved model
    saved_models = os.listdir('./models_saved')

    for param in model.parameters(): #frozen CLIP
        param.requires_grad = False

    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    if(dataset1 == 'CIFAR10'):
        cifar10_classes = get_cifar10_classes()
    elif(dataset1 == 'CIFAR100'):
        cifar100_classes = get_cifar100_classes()
    elif(dataset1 == 'ImageNet'):
        ImageNet_classes = imagenet_classes
    if(dataset2 == 'INbreast' or dataset3 == 'INbreast'):
        INbreast_classes = iNbreast_classes
    if(dataset2 == 'SIIMACR' or dataset3 == 'SIIMACR'):
        SIIMACR_classes = sIIMACR_classes

    if('CIFAR' in dataset1):
        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    elif('ImageNet' in dataset1):
        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)

    if('INbreast' in dataset2 or 'INbreast' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.INbreast_path, preprocess = preprocess)
        INbreast_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(INbreast_loader)
        print('INbreast dataset_size ', dataset_size)
    if('SIIMACR' in dataset2 or 'SIIMACR' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.SIIMACR_path, preprocess = preprocess)
        SIIMACR_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(SIIMACR_loader)
        print('SIIMACR dataset_size ', dataset_size)
    if('MIMICGAZE' in dataset1):
        dataset = CustomedMedicalData(data_root= opt.MIMICGAZE_path, preprocess = preprocess)
        MIMICGAZE_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        tra_loader = MIMICGAZE_loader
        dataset_size = len(MIMICGAZE_loader)
        print('MIMICGAZE dataset_size ', dataset_size)
    if('CheXpert5x200' in dataset4):
        dataset = CustomedMedicalData(data_root= opt.CheXpert5x200_path, preprocess = preprocess)
        CheXpert5x200_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(CheXpert5x200_loader)
        print('CheXpert5x200 dataset_size ', dataset_size)


    #Finetune
    phase = "train"
    for epoch in range( opt.epochs):
        total_loss = 0
        batch_num = 0
        #model.train()
        cp_network.train()
        for images,label_tokens,labels, heatmaps in tra_loader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            with torch.set_grad_enabled(phase == "train"):
                #logits_per_image, logits_per_text = model(images, label_tokens)
                image_features = model.encode_image(images)
                text_features = model.encode_text(label_tokens)
                # normalized features
                #image_features = image_features / image_features.norm(dim=1, keepdim=True)
                #text_features = text_features / text_features.norm(dim=1, keepdim=True)

                ###################################
                #CP Network acts on imges features and text features
                image_features = cp_network(image_features)
                text_features = cp_network(text_features)
                ###################################

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                total_loss = total_loss + cur_loss
                if phase == "train":
                    cur_loss.backward()  #retain_graph=True
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        #clip.model.convert_weights(model)  #basically convert the CLIP model weight into float16.

            if batch_num % 5 == 0:
                logger.info('{} epoch:{} loss_clip:{}'.format(phase,epoch,cur_loss))
                #break   # !!!!!!!!debug purpose, pls comment this after debug.
        epoch_clip_loss = total_loss / batch_num
            #torch.save(model.state_dict(),f"{model_name}_epoch_{epoch}.pth")
            #logger.info(f"weights_{epoch} saved")
            #logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

        scheduler.step()

        model.eval()    
        cp_network.eval()
        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in SIIMACR_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in SIIMACR_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(SIIMACR_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((SIIMACR_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('SIIMACR acc: ', correct/total)
            acc = round(correct/total, 4)
            if(acc> SIIMACR_stored_acc):
                SIIMACR_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' not in temp_file and 'finetune' in temp_file and 'SIIMACR'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(SIIMACR_stored_acc)}_{model_name}_zeroshoton_SIIMACR_{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")


        correct = 0
        total = 0
        normal_correct = 0
        benign_correct = 0
        malignant_correct = 0
        normal_total = 0
        benign_total = 0
        malignant_total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in INbreast_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in INbreast_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(INbreast_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((INbreast_classes[indices[0]] == labels[img_cnt]))

                    if(labels[img_cnt] == 'normal'):    #normal', 'benign', 'malignant'
                        normal_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            normal_correct += 1
                    if(labels[img_cnt] == 'benign'):    #normal', 'benign', 'malignant'
                        benign_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            benign_correct += 1
                    if(labels[img_cnt] == 'malignant'):    #normal', 'benign', 'malignant'
                        malignant_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            malignant_correct += 1

                    img_cnt += 1
            print('INbreast acc: ', correct/total)
            temp_normal_acc = round(normal_correct / normal_total, 2)  #round(answer, 2)
            temp_benign_acc = round(benign_correct / benign_total, 2)
            temp_malignant_acc = round(malignant_correct / malignant_total, 2)
            acc = round(correct/total, 4)
            if(acc> INbreast_stored_acc):
                INbreast_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' not in temp_file and 'finetune' in temp_file and 'INbreast'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(temp_normal_acc)}_{str(temp_benign_acc)}_{str(temp_malignant_acc)}_{str(INbreast_stored_acc)}_FrozenCLIP_{model_name}_zeroshoton_INbreast_{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")


        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in cheXpert_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in CheXpert5x200_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(cheXpert_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((cheXpert_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('CheXpert acc: ', correct/total)
            acc = round(correct/total, 4)
            if(acc> CheXpert5x200_stored_acc):
                CheXpert5x200_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' not in temp_file and 'finetune' in temp_file and 'CheXpert5x200'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(CheXpert5x200_stored_acc)}_{model_name}_zeroshoton_CheXpert5x200_{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")




###############################################################################################

def ZeroShotCLIP(opt_dataset):
    model_name = "Ori_CLIPZeroShoton_"+opt_dataset
    model, preprocess = clip.load(opt.vision_encoder, device=device,jit=False)
    #input_resolution = model.visual.input_resolution
    #context_length = model.context_length
    #vocab_size = model.vocab_size

    '''
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    '''

    if(opt_dataset == 'CIFAR10'):
        classes = get_cifar10_classes()
    elif(opt_dataset == 'CIFAR100'):
        classes = get_cifar100_classes()
    elif(opt_dataset == 'ImageNet'):
        classes = imagenet_classes
    elif(opt_dataset == 'INbreast'):
        classes = iNbreast_classes
    elif(opt_dataset == 'SIIMACR'):
        classes = sIIMACR_classes
    elif(opt_dataset == 'CheXpert5x200'):
        classes = cheXpert_classes
    len_classes = len(classes)

    if('CIFAR' in opt_dataset):
        dataset = CustomedData(img_root= './data'+'/'+opt_dataset, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedData(img_root= './data'+'/'+opt_dataset, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    elif('ImageNet' in opt_dataset):
        dataset = CustomedImageNet(img_root= './data'+'/'+opt_dataset, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedImageNet(img_root= './data'+'/'+opt_dataset, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    elif('INbreast' in opt_dataset):
        dataset = CustomedMedicalData(data_root= opt.INbreast_path, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('dataset_size ', dataset_size)
    elif('SIIMACR' in opt_dataset):
        dataset = CustomedMedicalData(data_root= opt.SIIMACR_path, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('dataset_size ', dataset_size)
    elif('MIMICGAZE' in opt_dataset):
        dataset = CustomedMedicalData(data_root= opt.MIMICGAZE_path, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('dataset_size ', dataset_size)
    elif('CheXpert5x200' in opt_dataset):
        dataset = CustomedMedicalData(data_root= opt.CheXpert5x200_path, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('dataset_size ', dataset_size)


    model.eval()    
    correct = 0
    total = 0
    stored_acc = 0
    text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in classes]).to(device)
    with torch.no_grad():
        for images, label_tokens,labels in test_loader:
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            img_cnt = 0
            for image_input in images:
                image_features = model.encode_image(image_input.unsqueeze(0))
                text_features = model.encode_text(text_inputs)
                # Pick the top 5 most similar labels for the image
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(len_classes) 
                # Print the result
                #print("\nTop predictions:\n")
                #for value, index in zip(values, indices):
                #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                total += 1
                correct += int((classes[indices[0]] == labels[img_cnt]))
                img_cnt += 1
        
        acc = round(correct/total, 4)

        if(acc> stored_acc):
            stored_acc = acc
            temp_files = os.listdir(opt.model_saved_path)
            #for temp_file in temp_files:
            #    if('CLIPZeroShot' in temp_file and model_name in temp_file and opt_dataset==temp_file.split('_')[-4]):
            #        os.remove(opt.model_saved_path+'/'+temp_file)
            checkpoint_path = f"{opt.vision_encoder}_{str(stored_acc)}_{model_name}_x_x_x.pth"
            checkpoint = {
                'network': model.state_dict()}
            torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
            logger.info(f"checkpoint saved")

###############################################################################################

def FineTuneCLIP(dataset1, dataset2, dataset3, dataset4):

    model, preprocess = clip.load(opt.vision_encoder, device=device,jit=False)
    #input_resolution = model.visual.input_resolution
    #context_length = model.context_length
    #vocab_size = model.vocab_size

    #for param in model.parameters(): #frozen CLIP
    #    param.requires_grad = False
    # Verify the trainable status of each parameter
    
    for name, param in model.named_parameters():
        if('visual' not in name):
            param.requires_grad = False
        #print(f"{name}: {param.requires_grad}")
    

    '''
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    '''

    #print(preprocess)
    #print(model)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr_clip,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
    #scheduler = lr_scheduler.StepLR(
    #        optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-10, last_epoch=- 1, verbose=False)
    
    
    inbreast_stored_acc = 0
    siim_stored_acc = 0
    chexpert5x200_stored_acc = 0
    '''
    #load saved model
    saved_models = os.listdir('./models_saved')
    for saved_model in saved_models:
        if( 'finetune' in saved_model and 'Ori_CLIP' in saved_model and dataset1==saved_model.split('_')[-4]):
            checkpoint = torch.load('./models_saved/'+saved_model)
            model.load_state_dict(checkpoint['network'])
            print('load saved models done!')
            stored_acc = float(saved_model.split('_')[0])
    '''

    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    if(dataset1 == 'CIFAR10'):
        cifar10_classes = get_cifar10_classes()
    elif(dataset1 == 'CIFAR100'):
        cifar100_classes = get_cifar100_classes()
    elif(dataset1 == 'ImageNet'):
        ImageNet_classes = imagenet_classes
    if(dataset2 == 'INbreast' or dataset3 == 'INbreast'):
        INbreast_classes = iNbreast_classes
    if(dataset2 == 'SIIMACR' or dataset3 == 'SIIMACR'):
        SIIMACR_classes = sIIMACR_classes
    

    if('CIFAR' in dataset1):
        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    elif('ImageNet' in dataset1):
        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    
    if('INbreast' in dataset2 or 'INbreast' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.INbreast_path, preprocess = preprocess)
        INbreast_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(INbreast_loader)
        print('INbreast dataset_size ', dataset_size)
    if('SIIMACR' in dataset2 or 'SIIMACR' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.SIIMACR_path, preprocess = preprocess)
        SIIMACR_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(SIIMACR_loader)
        print('SIIMACR dataset_size ', dataset_size)
    if('MIMICGAZE' in dataset1):
        dataset = CustomedMedicalData(data_root= opt.MIMICGAZE_path, preprocess = preprocess)
        MIMICGAZE_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        tra_loader = MIMICGAZE_loader
        dataset_size = len(MIMICGAZE_loader)
        print('MIMICGAZE dataset_size ', dataset_size)
    if('CheXpert5x200' in dataset4):
        dataset = CustomedMedicalData(data_root= opt.CheXpert5x200_path, preprocess = preprocess)
        CheXpert5x200_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(CheXpert5x200_loader)
        print('CheXpert5x200 dataset_size ', dataset_size)


    #Finetune
    phase = "train"

    for epoch in range( opt.epochs):
        print('epoch ', epoch)
        total_loss = 0
        batch_num = 0
        model.train()
        # 使用混合精度，占用显存更小
        with torch.cuda.amp.autocast(enabled=True):
            for images,label_tokens,labels, heatmaps in tra_loader:
                # 将图片和标签token转移到device设备
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                batch_num += 1
                # 优化器梯度清零
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    logits_per_image, logits_per_text = model(images, label_tokens)
                    #image_features = model.encode_image(images)   
                    #text_features = model.encode_text(label_tokens)
                    ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
                    cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
                    total_loss += cur_loss
                    if phase == "train":
                        cur_loss.backward()
                        if device == "cpu":
                            optimizer.step()
                        else:
                            optimizer.step()
                            #clip.model.convert_weights(model)

                if batch_num % 5 == 0:
                    logger.info('{} epoch:{} loss:{}'.format(phase,epoch,cur_loss))
                #break   # !!!!!!!!debug purpose, pls comment this after debug.
            epoch_loss = total_loss / batch_num
            #torch.save(model.state_dict(),f"{model_name}_epoch_{epoch}.pth")
            #logger.info(f"weights_{epoch} saved")
            #logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
        
        scheduler.step()

        model.eval()    
        
        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in INbreast_classes]).to(device)
        normal_correct = 0
        benign_correct = 0
        malignant_correct = 0
        normal_total = 0
        benign_total = 0
        malignant_total = 0
        with torch.no_grad():
            for images, label_tokens,labels in INbreast_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(INbreast_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((INbreast_classes[indices[0]] == labels[img_cnt]))

                    if(labels[img_cnt] == 'normal'):    #normal', 'benign', 'malignant'
                        normal_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            normal_correct += 1
                    if(labels[img_cnt] == 'benign'):    #normal', 'benign', 'malignant'
                        benign_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            benign_correct += 1
                    if(labels[img_cnt] == 'malignant'):    #normal', 'benign', 'malignant'
                        malignant_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            malignant_correct += 1

                    img_cnt += 1
            print('inbreast acc: ', correct/total)
            temp_normal_acc = round(normal_correct / normal_total, 2)
            temp_benign_acc = round(benign_correct / benign_total, 2)
            temp_malignant_acc = round(malignant_correct / malignant_total, 2)
            acc = round(correct/total, 4)
            if(acc> inbreast_stored_acc):
                inbreast_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('finetune' in temp_file and 'Ori_CLIP' in temp_file and dataset1==temp_file.split('_')[-6] and 'INbreast'==temp_file.split('_')[-4] ):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(temp_normal_acc)}_{str(temp_benign_acc)}_{str(temp_malignant_acc)}_{str(inbreast_stored_acc)}_Ori_CLIPFinetunedOn_{dataset1}_ZeroShoton_INbreast_x_x_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")
        

        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in SIIMACR_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in SIIMACR_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(SIIMACR_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((SIIMACR_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('siim acc: ', correct/total)
            acc = round(correct/total, 4)
            if(acc> siim_stored_acc):
                siim_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('finetune' in temp_file and 'Ori_CLIP' in temp_file and dataset1==temp_file.split('_')[-6] and 'SIIMACR'==temp_file.split('_')[-4] ):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(siim_stored_acc)}_Ori_CLIPFinetunedOn_{dataset1}_ZeroShoton_SIIMACR_x_x_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")
        

        
        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in cheXpert_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in CheXpert5x200_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(cheXpert_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((cheXpert_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('cheXpert acc: ', correct/total)
            acc = round(correct/total, 4)
            if(acc> chexpert5x200_stored_acc):
                chexpert5x200_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('finetune' in temp_file and 'Ori_CLIP' in temp_file and dataset1==temp_file.split('_')[-6] and 'CheXpert5x200'==temp_file.split('_')[-4] ):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(chexpert5x200_stored_acc)}_Ori_CLIPFinetunedOn_{dataset1}_ZeroShoton_CheXpert5x200_x_x_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")
        

def FineTuneCPCLIP_GAZE(dataset1, dataset2, dataset3, dataset4):
    model_name = "GAZE_CP_CLIPFinetunedOn_"+dataset1
    #torch.autograd.set_detect_anomaly(True)

    model, preprocess = clip.load(opt.vision_encoder, device=device,jit=False)
    #input_resolution = model.visual.input_resolution
    #context_length = model.context_length
    #vocab_size = model.vocab_size
    #logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    
    for name, param in model.named_parameters():
        if('visual' not in name):
            param.requires_grad = False
        #print(f"{name}: {param.requires_grad}")
    

    #add Core-periphery NN after CLIP
    if('RN' in opt.vision_encoder):
        nodes = nodes_lookup_table['RN']
        cores = round(nodes * opt.ratio)
    elif('ViT' in opt.vision_encoder):
        nodes = nodes_lookup_table['ViT']
        cores = round(nodes * opt.ratio)
    cp_mask = get_cp_mask(nodes, cores)
    cp_network = CP_NN(in_dim=nodes, out_dim=nodes, CP_mask= torch.from_numpy(cp_mask).to(device), device=device)
    cp_network.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('clip params ', params/1e6)

    '''
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    '''

    #print(preprocess)
    #print(model)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr':opt.lr_clip},
                           {'params': cp_network.parameters(), 'lr': opt.lr_cpNN}], 
                           betas=(0.9,0.99),eps=1e-6,weight_decay=0.001)

    #scheduler = lr_scheduler.StepLR(
    #       optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-10, last_epoch=- 1, verbose=False)

    SIIMACR_stored_acc = 0
    INbreast_stored_acc = 0
    CheXpert5x200_stored_acc = 0
    #load saved model
    saved_models = os.listdir('./models_saved')
        
    print('in Finetune CP CLIP')

    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    if(dataset1 == 'CIFAR10'):
        cifar10_classes = get_cifar10_classes()
    elif(dataset1 == 'CIFAR100'):
        cifar100_classes = get_cifar100_classes()
    elif(dataset1 == 'ImageNet'):
        ImageNet_classes = imagenet_classes
    if(dataset2 == 'INbreast' or dataset3 == 'INbreast'):
        INbreast_classes = iNbreast_classes
    if(dataset2 == 'SIIMACR' or dataset3 == 'SIIMACR'):
        SIIMACR_classes = sIIMACR_classes

    if('CIFAR' in dataset1):
        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    elif('ImageNet' in dataset1):
        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)

    if('INbreast' in dataset2 or 'INbreast' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.INbreast_path, preprocess = preprocess)
        INbreast_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(INbreast_loader)
        print('INbreast dataset_size ', dataset_size)
    if('SIIMACR' in dataset2 or 'SIIMACR' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.SIIMACR_path, preprocess = preprocess)
        SIIMACR_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(SIIMACR_loader)
        print('SIIMACR dataset_size ', dataset_size)
    if('MIMICGAZE' in dataset1):
        dataset = CustomedMedicalData(data_root= opt.MIMICGAZE_path, preprocess = preprocess)
        MIMICGAZE_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        tra_loader = MIMICGAZE_loader
        dataset_size = len(MIMICGAZE_loader)
        print('MIMICGAZE dataset_size ', dataset_size)
    if('CheXpert5x200' in dataset4):
        dataset = CustomedMedicalData(data_root= opt.CheXpert5x200_path, preprocess = preprocess)
        CheXpert5x200_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(CheXpert5x200_loader)
        print('CheXpert5x200 dataset_size ', dataset_size)


    #Finetune
    phase = "train"
    entropylossfn = nn.CosineSimilarity()
    for epoch in range( opt.epochs):
        total_loss = 0
        batch_num = 0
        model.train()
        cp_network.train()
        for images,label_tokens,labels, heatmaps in tra_loader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            heatmaps= heatmaps.to(device)
            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            with torch.set_grad_enabled(phase == "train"):
                #logits_per_image, logits_per_text = model(images, label_tokens)
                image_features = model.encode_image(images)
                text_features = model.encode_text(label_tokens)  #ViT dim is 512, ResNet dim is 1024

                heatmap_features = model.encode_image(heatmaps)

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                ###################################
                #CP Network acts on imges features and text features
                image_features = cp_network(image_features)
                text_features = cp_network(text_features)

                heatmap_features = cp_network(heatmap_features)
                ###################################

                # normalized features
                #image_features = image_features / image_features.norm(dim=1, keepdim=True)
                #text_features = text_features / text_features.norm(dim=1, keepdim=True)

                heatmap_features = heatmap_features / heatmap_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                entropyloss = torch.mean(entropylossfn(heatmap_features, image_features))
                cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2-entropyloss

                total_loss = total_loss + cur_loss
                if phase == "train":
                    (cur_loss).backward()  #retain_graph=True
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        #clip.model.convert_weights(model)

            if batch_num % 3 == 0:
                logger.info('{} epoch:{} loss_clip:{} entropyloss {}'.format(phase,epoch,cur_loss, entropyloss))
                #break   # !!!!!!!!debug purpose, pls comment this after debug.
        epoch_clip_loss = total_loss / batch_num
            #torch.save(model.state_dict(),f"{model_name}_epoch_{epoch}.pth")
            #logger.info(f"weights_{epoch} saved")
            #logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))
        
        scheduler.step()

        model.eval()    
        
        cp_network.eval()
        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in SIIMACR_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in SIIMACR_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(SIIMACR_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((SIIMACR_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('SIIMACR acc is ', correct/total)
            acc = round(correct/total, 4)
            if(acc> SIIMACR_stored_acc):
                SIIMACR_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' not in temp_file and 'finetune' in temp_file and 'SIIMACR'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and 'FrozenCLIP' not in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(SIIMACR_stored_acc)}_{model_name}_zeroshot_on_SIIMACR_{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        correct = 0
        total = 0
        normal_correct = 0
        benign_correct = 0
        malignant_correct = 0
        normal_total = 0
        benign_total = 0
        malignant_total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in INbreast_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in INbreast_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(INbreast_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((INbreast_classes[indices[0]] == labels[img_cnt]))

                    if(labels[img_cnt] == 'normal'):    #normal', 'benign', 'malignant'
                        normal_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            normal_correct += 1
                    if(labels[img_cnt] == 'benign'):    #normal', 'benign', 'malignant'
                        benign_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            benign_correct += 1
                    if(labels[img_cnt] == 'malignant'):    #normal', 'benign', 'malignant'
                        malignant_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            malignant_correct += 1

                    img_cnt += 1
            print('INbreast acc is ', correct/total)
            temp_normal_acc = round(normal_correct / normal_total, 2)
            temp_benign_acc = round(benign_correct / benign_total, 2)
            temp_malignant_acc = round(malignant_correct / malignant_total, 2)
            acc = round(correct/total, 4)
            if(acc> INbreast_stored_acc):
                INbreast_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' not in temp_file and 'finetune' in temp_file and 'INbreast'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and 'FrozenCLIP' not in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(temp_normal_acc)}_{str(temp_benign_acc)}_{str(temp_malignant_acc)}_{str(INbreast_stored_acc)}_{model_name}_zeroshot_on_INbreast__{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        
        correct = 0
        total = 0
        temp_cardi = 0
        temp_edema = 0
        temp_consol = 0 
        temp_atelect = 0
        temp_pleural = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in cheXpert_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in CheXpert5x200_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(cheXpert_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((cheXpert_classes[indices[0]] == labels[img_cnt]))
                    if(cheXpert_classes[indices[0]] == 'cardiomegaly'):
                        temp_cardi +=1
                    elif(cheXpert_classes[indices[0]] == 'edema'):
                        temp_edema += 1
                    elif(cheXpert_classes[indices[0]] == 'consolidation'):
                        temp_consol += 1
                    elif(cheXpert_classes[indices[0]] == 'atelectasis'):
                        temp_atelect += 1
                    elif(cheXpert_classes[indices[0]] == 'pleural effusion'):
                        temp_pleural += 1
                    img_cnt += 1
            print('cardimegaly {}, edema {}, consolidation {}, atelectasis {}, pleural effusion {}'.format(temp_cardi, temp_edema, temp_consol, temp_atelect, temp_pleural))
            print('CheXpert acc is ', correct/total)
            acc = round(correct/total, 4)
            if(acc> CheXpert5x200_stored_acc):
                CheXpert5x200_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' not in temp_file and 'finetune' in temp_file and 'CheXpert5x200'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and 'FrozenCLIP' not in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(CheXpert5x200_stored_acc)}_{model_name}_zeroshot_on_CheXpert5x200_{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

def FineTuneCPCLIP_FrozenCLIP_GAZE(dataset1, dataset2, dataset3, dataset4):
    model_name = "GAZE_FineTuneCPCLIP_FrozenCLIPOn_"+dataset1
    #torch.autograd.set_detect_anomaly(True)

    model, preprocess = clip.load(opt.vision_encoder, device=device,jit=False)
    #input_resolution = model.visual.input_resolution
    #context_length = model.context_length
    #vocab_size = model.vocab_size
    #logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    #add Core-periphery NN after CLIP
    if('RN' in opt.vision_encoder):
        nodes = nodes_lookup_table['RN']
        cores = round(nodes * opt.ratio)
    elif('ViT' in opt.vision_encoder):
        nodes = nodes_lookup_table['ViT']
        cores = round(nodes * opt.ratio)
    cp_mask = get_cp_mask(nodes, cores)
    cp_network = CP_NN(in_dim=nodes, out_dim=nodes, CP_mask= torch.from_numpy(cp_mask).to(device), device=device)
    cp_network.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('clip params ', params/1e6)

    '''
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    '''

    #print(preprocess)
    #print(model)
    optimizer = optim.Adam([
                           {'params': cp_network.parameters(), 'lr': opt.lr_cpNN_fzCLIP}], 
                           betas=(0.9,0.99),eps=1e-6,weight_decay=0.001)

    #scheduler = lr_scheduler.StepLR(
    #       optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-10, last_epoch=- 1, verbose=False)

    SIIMACR_stored_acc = 0
    INbreast_stored_acc = 0
    CheXpert5x200_stored_acc = 0
    #load saved model
    saved_models = os.listdir('./models_saved')

    for param in model.parameters(): #frozen CLIP
        param.requires_grad = False

    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_embedsimi = nn.CosineSimilarity()

    if(dataset1 == 'CIFAR10'):
        cifar10_classes = get_cifar10_classes()
    elif(dataset1 == 'CIFAR100'):
        cifar100_classes = get_cifar100_classes()
    elif(dataset1 == 'ImageNet'):
        ImageNet_classes = imagenet_classes
    if(dataset2 == 'INbreast' or dataset3 == 'INbreast'):
        INbreast_classes = iNbreast_classes
    if(dataset2 == 'SIIMACR' or dataset3 == 'SIIMACR'):
        SIIMACR_classes = sIIMACR_classes

    if('CIFAR' in dataset1):
        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    elif('ImageNet' in dataset1):
        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)

    if('INbreast' in dataset2 or 'INbreast' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.INbreast_path, preprocess = preprocess)
        INbreast_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(INbreast_loader)
        print('INbreast dataset_size ', dataset_size)
    if('SIIMACR' in dataset2 or 'SIIMACR' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.SIIMACR_path, preprocess = preprocess)
        SIIMACR_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(SIIMACR_loader)
        print('SIIMACR dataset_size ', dataset_size)
    if('MIMICGAZE' in dataset1):
        dataset = CustomedMedicalData(data_root= opt.MIMICGAZE_path, preprocess = preprocess)
        MIMICGAZE_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        tra_loader = MIMICGAZE_loader
        dataset_size = len(MIMICGAZE_loader)
        print('MIMICGAZE dataset_size ', dataset_size)
    if('CheXpert5x200' in dataset4):
        dataset = CustomedMedicalData(data_root= opt.CheXpert5x200_path, preprocess = preprocess)
        CheXpert5x200_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(CheXpert5x200_loader)
        print('CheXpert5x200 dataset_size ', dataset_size)


    #Finetune
    phase = "train"
    for epoch in range( opt.epochs):
        total_loss = 0
        batch_num = 0
        #model.train()
        cp_network.train()
        for images,label_tokens,labels, heatmaps in tra_loader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)

            heatmaps = heatmaps.to(device)

            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            with torch.set_grad_enabled(phase == "train"):
                #logits_per_image, logits_per_text = model(images, label_tokens)
                image_features = model.encode_image(images)
                text_features = model.encode_text(label_tokens)

                heatmap_features = model.encode_image(heatmaps)
                # normalized features
                #image_features = image_features / image_features.norm(dim=1, keepdim=True)
                #text_features = text_features / text_features.norm(dim=1, keepdim=True)

                ###################################
                #CP Network acts on imges features and text features
                image_features = cp_network(image_features)
                text_features = cp_network(text_features)

                heatmap_features = cp_network(heatmap_features)
                ###################################

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                heatmap_features = heatmap_features / heatmap_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

                img_heatmap_simi_loss = loss_embedsimi(image_features, heatmap_features)

                total_loss = total_loss + cur_loss - img_heatmap_simi_loss
                if phase == "train":
                    (cur_loss - img_heatmap_simi_loss).backward()  #retain_graph=True
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        #clip.model.convert_weights(model)  #basically convert the CLIP model weight into float16.

            if batch_num % 5 == 0:
                logger.info('{} epoch:{} loss_clip:{} img_heatmap_simi_loss {}'.format(phase,epoch,cur_loss, img_heatmap_simi_loss))
                #break   # !!!!!!!!debug purpose, pls comment this after debug.
        epoch_clip_loss = total_loss / batch_num
            #torch.save(model.state_dict(),f"{model_name}_epoch_{epoch}.pth")
            #logger.info(f"weights_{epoch} saved")
            #logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

        scheduler.step()

        model.eval()    
        cp_network.eval()
        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in SIIMACR_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in SIIMACR_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(SIIMACR_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((SIIMACR_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('SIIMACR acc: ', correct/total)
            acc = round(correct/total, 4)
            if(acc> SIIMACR_stored_acc):
                SIIMACR_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' not in temp_file and 'finetune' in temp_file and 'SIIMACR'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(SIIMACR_stored_acc)}_FrozenCLIP_{model_name}_zeroshoton_SIIMACR_{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")


        correct = 0
        total = 0
        normal_correct = 0
        benign_correct = 0
        malignant_correct = 0
        normal_total = 0
        benign_total = 0
        malignant_total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in INbreast_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in INbreast_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(INbreast_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((INbreast_classes[indices[0]] == labels[img_cnt]))

                    if(labels[img_cnt] == 'normal'):    #normal', 'benign', 'malignant'
                        normal_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            normal_correct += 1
                    if(labels[img_cnt] == 'benign'):    #normal', 'benign', 'malignant'
                        benign_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            benign_correct += 1
                    if(labels[img_cnt] == 'malignant'):    #normal', 'benign', 'malignant'
                        malignant_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            malignant_correct += 1

                    img_cnt += 1
            print('INbreast acc: ', correct/total)
            temp_normal_acc = round(normal_correct / normal_total, 2)  #round(answer, 2)
            temp_benign_acc = round(benign_correct / benign_total, 2)
            temp_malignant_acc = round(malignant_correct / malignant_total, 2)
            acc = round(correct/total, 4)
            if(acc> INbreast_stored_acc):
                INbreast_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' not in temp_file and 'finetune' in temp_file and 'INbreast'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(temp_normal_acc)}_{str(temp_benign_acc)}_{str(temp_malignant_acc)}_{str(INbreast_stored_acc)}_FrozenCLIP_{model_name}_zeroshoton_INbreast_{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")


        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in cheXpert_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in CheXpert5x200_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(cheXpert_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((cheXpert_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('CheXpert acc: ', correct/total)
            acc = round(correct/total, 4)
            if(acc> CheXpert5x200_stored_acc):
                CheXpert5x200_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' not in temp_file and 'finetune' in temp_file and 'CheXpert5x200'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(CheXpert5x200_stored_acc)}_FrozenCLIP_{model_name}_zeroshoton_CheXpert5x200_{str(nodes)}_{str(cores)}_finetune.pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

def FineTuneCPCLIPv2_GAZE(dataset1, dataset2, dataset3, dataset4):
    model_name = "GAZE_CP_CLIPFinetunedv2On_"+dataset1
    #different from FineTuneCLIPv2 that take information from cores and peripheries nodes; this version only take information from core nodes
    #torch.autograd.set_detect_anomaly(True)

    model, preprocess = clip.load(opt.vision_encoder, device=device,jit=False)
    #input_resolution = model.visual.input_resolution
    #context_length = model.context_length
    #vocab_size = model.vocab_size
    #logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
     
    
    for name, param in model.named_parameters():
        if('visual' not in name):
            param.requires_grad = False
        #print(f"{name}: {param.requires_grad}")
    

    #add Core-periphery NN after CLIP
    if('RN' in opt.vision_encoder):
        nodes = nodes_lookup_table['RN']
        cores = round(nodes * opt.ratio)
    elif('ViT' in opt.vision_encoder):
        nodes = nodes_lookup_table['ViT']
        cores = round(nodes * opt.ratio)
    cp_mask = get_cp_mask(nodes, cores)
    cp_network = CP_NN(in_dim=nodes, out_dim=nodes, CP_mask= torch.from_numpy(cp_mask).to(device), device=device)
    cp_network.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('clip params ', params/1e6)

    '''
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    '''

    #print(preprocess)
    #print(model)
    optimizer = optim.Adam([{'params': model.parameters(), 'lr':opt.lr_clip},
                           {'params': cp_network.parameters(), 'lr': opt.lr_cpNN}], 
                           betas=(0.9,0.99),eps=1e-6,weight_decay=0.001)

    #scheduler = lr_scheduler.StepLR(
    #       optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-10, last_epoch=- 1, verbose=False)

    SIIMACR_stored_acc = 0
    INbreast_stored_acc = 0
    CheXpert_stored_acc = 0
    #load saved model
    saved_models = os.listdir('./models_saved')
        
    print('in FineTuneCPCLIPv2')

    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_embedsimi = nn.CosineSimilarity()

    if(dataset1 == 'CIFAR10'):
        cifar10_classes = get_cifar10_classes()
    elif(dataset1 == 'CIFAR100'):
        cifar100_classes = get_cifar100_classes()
    elif(dataset1 == 'ImageNet'):
        ImageNet_classes = imagenet_classes
    if(dataset2 == 'INbreast' or dataset3 == 'INbreast'):
        INbreast_classes = iNbreast_classes
    if(dataset2 == 'SIIMACR' or dataset3 == 'SIIMACR'):
        SIIMACR_classes = sIIMACR_classes

    if('CIFAR' in dataset1):
        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    elif('ImageNet' in dataset1):
        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)

    if('INbreast' in dataset2 or 'INbreast' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.INbreast_path, preprocess = preprocess)
        INbreast_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(INbreast_loader)
        print('INbreast dataset_size ', dataset_size)
    if('SIIMACR' in dataset2 or 'SIIMACR' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.SIIMACR_path, preprocess = preprocess)
        SIIMACR_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(SIIMACR_loader)
        print('SIIMACR dataset_size ', dataset_size)
    if('MIMICGAZE' in dataset1):
        dataset = CustomedMedicalData(data_root= opt.MIMICGAZE_path, preprocess = preprocess)
        MIMICGAZE_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        tra_loader = MIMICGAZE_loader
        dataset_size = len(MIMICGAZE_loader)
        print('MIMICGAZE dataset_size ', dataset_size)
    if('CheXpert5x200' in dataset4):
        dataset = CustomedMedicalData(data_root= opt.CheXpert5x200_path, preprocess = preprocess)
        CheXpert5x200_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(CheXpert5x200_loader)
        print('CheXpert5x200 dataset_size ', dataset_size)


    #Finetune
    phase = "train"

    for epoch in range( opt.epochs):
        total_loss = 0
        batch_num = 0
        model.train()
        cp_network.train()
        for images,label_tokens,labels, heatmaps in tra_loader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)
            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            with torch.set_grad_enabled(phase == "train"):
                #logits_per_image, logits_per_text = model(images, label_tokens)
                image_features = model.encode_image(images)
                text_features = model.encode_text(label_tokens)

                heatmap_features = model.encode_image(heatmaps)
                # normalized features
                #image_features = image_features / image_features.norm(dim=1, keepdim=True)
                #text_features = text_features / text_features.norm(dim=1, keepdim=True)

                ###################################
                #CP Network acts on imges features and text features
                image_features = cp_network(image_features)
                text_features = cp_network(text_features)

                heatmap_features = cp_network(heatmap_features)
                ###################################

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                heatmap_features = heatmap_features / heatmap_features.norm(dim=1, keepdim=True)
                
                #take the features in cores
                image_features = image_features[:,0:cores]
                text_features = text_features[:,0:cores]

                # cosine similarity as logits
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

                img_heatmap_simi_loss = loss_embedsimi(image_features, heatmap_features)

                total_loss = total_loss + cur_loss - img_heatmap_simi_loss
                if phase == "train":
                    (cur_loss-img_heatmap_simi_loss).backward()  #retain_graph=True
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        #clip.model.convert_weights(model)  #basically convert the CLIP model weight into float16.

            if batch_num % 5 == 0:
                logger.info('{} epoch:{} loss_clip:{} img_heatmap_simi_loss {}'.format(phase,epoch,cur_loss, img_heatmap_simi_loss))
                #break   # !!!!!!!!debug purpose, pls comment this after debug.
        epoch_clip_loss = total_loss / batch_num
            #torch.save(model.state_dict(),f"{model_name}_epoch_{epoch}.pth")
            #logger.info(f"weights_{epoch} saved")
            #logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

        scheduler.step()
 
        model.eval()    
        cp_network.eval()
        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in SIIMACR_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in SIIMACR_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(SIIMACR_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((SIIMACR_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('SIIMACR acc: ', correct/total)
            acc = round(correct/total, 4)
            if(acc> SIIMACR_stored_acc):
                SIIMACR_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' in temp_file and 'finetune' in temp_file and 'SIIMACR'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and str(nodes)==temp_file.split('_')[-3] and 'FrozenCLIP' not in temp_file and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(SIIMACR_stored_acc)}_{model_name}_zeroshoton_SIIMACR_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        correct = 0
        total = 0
        normal_correct = 0
        benign_correct = 0
        malignant_correct = 0
        normal_total = 0
        benign_total = 0
        malignant_total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in INbreast_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in INbreast_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(INbreast_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((INbreast_classes[indices[0]] == labels[img_cnt]))

                    if(labels[img_cnt] == 'normal'):    #normal', 'benign', 'malignant'
                        normal_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            normal_correct += 1
                    if(labels[img_cnt] == 'benign'):    #normal', 'benign', 'malignant'
                        benign_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            benign_correct += 1
                    if(labels[img_cnt] == 'malignant'):    #normal', 'benign', 'malignant'
                        malignant_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            malignant_correct += 1

                    img_cnt += 1
            print('INbreast acc: ', correct/total)
            temp_normal_acc = round(normal_correct / normal_total, 2)
            temp_benign_acc = round(benign_correct / benign_total, 2)
            temp_malignant_acc = round(malignant_correct / malignant_total, 2)
            acc = round(correct/total, 4)
            if(acc> INbreast_stored_acc):
                INbreast_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' in temp_file and 'finetune' in temp_file and 'INbreast'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and str(nodes)==temp_file.split('_')[-3] and 'FrozenCLIP' not in temp_file and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(temp_normal_acc)}_{str(temp_benign_acc)}_{str(temp_malignant_acc)}_{str(INbreast_stored_acc)}_{model_name}_zeroshoton_INbreast_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in cheXpert_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in CheXpert5x200_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(cheXpert_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((cheXpert_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('CheXpert acc: ', correct/total)
            acc = round(correct/total, 4)
            if(acc> CheXpert_stored_acc):
                CheXpert_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('v2' in temp_file and 'finetune' in temp_file and 'CheXPert5x200'==temp_file.split('_')[-4] and 'CP_CLIP' in temp_file \
                #        and str(nodes)==temp_file.split('_')[-3] and 'FrozenCLIP' not in temp_file and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(CheXpert_stored_acc)}_{model_name}_zeroshoton_CheXPert5x200_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'network': model.state_dict(),
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

def FineTuneCPCLIPv2_FrozenCLIP_GAZE(dataset1, dataset2, dataset3, dataset4):
    model_name = "GAZE_FineTuneCPCLIPv2_FrozenCLIPOn_"+dataset1
    #different from FineTuneCLIPv2 that take information from cores and peripheries nodes; this version only take information from core nodes
    #torch.autograd.set_detect_anomaly(True)

    model, preprocess = clip.load(opt.vision_encoder, device=device,jit=False)
    #input_resolution = model.visual.input_resolution
    #context_length = model.context_length
    #vocab_size = model.vocab_size
    #logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    #add Core-periphery NN after CLIP
    if('RN' in opt.vision_encoder):
        nodes = nodes_lookup_table['RN']
        cores = round(nodes * opt.ratio)
    elif('ViT' in opt.vision_encoder):
        nodes = nodes_lookup_table['ViT']
        cores = round(nodes * opt.ratio)
    cp_mask = get_cp_mask(nodes, cores)
    cp_network = CP_NN(in_dim=nodes, out_dim=nodes, CP_mask= torch.from_numpy(cp_mask).to(device), device=device)
    cp_network.to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('clip params ', params/1e6)

    '''
    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Input resolution:", input_resolution)
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)
    '''

    #print(preprocess)
    #print(model)
    optimizer = optim.Adam([
                           {'params': cp_network.parameters(), 'lr': opt.lr_cpNN_fzCLIP}], 
                           betas=(0.9,0.99),eps=1e-6,weight_decay=0.001)

    #scheduler = lr_scheduler.StepLR(
    #       optimizer, step_size=10, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=1e-10, last_epoch=- 1, verbose=False)

    SIIMACR_stored_acc = 0
    INbreast_stored_acc = 0
    CheXpert_stored_acc = 0
    #load saved model
    saved_models = os.listdir('./models_saved')
    print('in FinetuneCPCLIPv2 frozen CLIP!')    

    for param in model.parameters(): #frozen CLIP
        param.requires_grad = False

    # 创建损失函数
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    loss_embedsimi = nn.CosineSimilarity()

    if(dataset1 == 'CIFAR10'):
        cifar10_classes = get_cifar10_classes()
    elif(dataset1 == 'CIFAR100'):
        cifar100_classes = get_cifar100_classes()
    elif(dataset1 == 'ImageNet'):
        ImageNet_classes = imagenet_classes
    if(dataset2 == 'INbreast' or dataset3 == 'INbreast'):
        INbreast_classes = iNbreast_classes
    if(dataset2 == 'SIIMACR' or dataset3 == 'SIIMACR'):
        SIIMACR_classes = sIIMACR_classes

    if('CIFAR' in dataset1):
        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedData(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)
    elif('ImageNet' in dataset1):
        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = True, preprocess = preprocess)
        tra_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(tra_loader)
        print('train dataset_size ', dataset_size)

        dataset = CustomedImageNet(img_root= './data'+'/'+dataset1, is_train = False, preprocess = preprocess)
        test_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(test_loader)
        print('test dataset_size ', dataset_size)

    if('INbreast' in dataset2 or 'INbreast' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.INbreast_path, preprocess = preprocess)
        INbreast_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(INbreast_loader)
        print('INbreast dataset_size ', dataset_size)
    if('SIIMACR' in dataset2 or 'SIIMACR' in dataset3):
        dataset = CustomedMedicalData(data_root= opt.SIIMACR_path, preprocess = preprocess)
        SIIMACR_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(SIIMACR_loader)
        print('SIIMACR dataset_size ', dataset_size)
    if('MIMICGAZE' in dataset1):
        dataset = CustomedMedicalData(data_root= opt.MIMICGAZE_path, preprocess = preprocess)
        MIMICGAZE_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        tra_loader = MIMICGAZE_loader
        dataset_size = len(MIMICGAZE_loader)
        print('MIMICGAZE dataset_size ', dataset_size)
    if('CheXpert5x200' in dataset4):
        dataset = CustomedMedicalData(data_root= opt.CheXpert5x200_path, preprocess = preprocess)
        CheXpert5x200_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = opt.batch_size,shuffle= True)
        dataset_size = len(CheXpert5x200_loader)
        print('CheXpert5x200 dataset_size ', dataset_size)

    #Finetune
    phase = "train"

    for epoch in range( opt.epochs ):
        total_loss = 0
        batch_num = 0
        #model.train()
        cp_network.train()
        for images,label_tokens,labels, heatmaps in tra_loader:
            # 将图片和标签token转移到device设备
            images = images.to(device)
            label_tokens = label_tokens.to(device)

            heatmaps = heatmaps.to(device)
            batch_num += 1
            # 优化器梯度清零
            optimizer.zero_grad()
            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
            with torch.set_grad_enabled(phase == "train"):
                #logits_per_image, logits_per_text = model(images, label_tokens)
                image_features = model.encode_image(images)
                text_features = model.encode_text(label_tokens)

                heatmap_features = model.encode_image(heatmaps)

                # normalized features
                #image_features = image_features / image_features.norm(dim=1, keepdim=True)
                #text_features = text_features / text_features.norm(dim=1, keepdim=True)

                ###################################
                #CP Network acts on imges features and text features
                image_features = cp_network(image_features)
                text_features = cp_network(text_features)

                heatmap_features = cp_network(heatmap_features)
                ###################################

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                heatmap_features = heatmap_features / heatmap_features.norm(dim=1, keepdim=True)
                
                #take the features in cores
                image_features = image_features[:,0:cores]
                text_features = text_features[:,0:cores]

                # cosine similarity as logits
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                cur_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

                img_heatmap_simi_loss = loss_embedsimi(image_features, heatmap_features)
                total_loss = total_loss + cur_loss - img_heatmap_simi_loss
                if phase == "train":
                    (cur_loss - img_heatmap_simi_loss).backward()  #retain_graph=True
                    if device == "cpu":
                        optimizer.step()
                    else:
                        optimizer.step()
                        #clip.model.convert_weights(model)  #basically convert the CLIP model weight into float16.

            if batch_num % 5 == 0:
                logger.info('{} epoch:{} loss_clip:{} img_heatmap_simi_loss:{}'.format(phase,epoch,cur_loss, img_heatmap_simi_loss))
                #break   # !!!!!!!!debug purpose, pls comment this after debug.
        epoch_clip_loss = total_loss / batch_num
            #torch.save(model.state_dict(),f"{model_name}_epoch_{epoch}.pth")
            #logger.info(f"weights_{epoch} saved")
            #logger.info('{} Loss: {:.4f}'.format(phase, epoch_loss))

        scheduler.step()

        model.eval()    
        cp_network.eval()
        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in SIIMACR_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in SIIMACR_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(SIIMACR_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((SIIMACR_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('SIIMACR acc: ',correct/total)
            acc = round(correct/total, 4)
            if(acc> SIIMACR_stored_acc):
                SIIMACR_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' in temp_file and 'finetune' in temp_file and 'SIIMACR'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(SIIMACR_stored_acc)}_{model_name}_zeroshoton_SIIMACR_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        correct = 0
        total = 0
        normal_correct = 0
        benign_correct = 0
        malignant_correct = 0
        normal_total = 0
        benign_total = 0
        malignant_total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in INbreast_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in INbreast_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(INbreast_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((INbreast_classes[indices[0]] == labels[img_cnt]))

                    if(labels[img_cnt] == 'normal'):    #normal', 'benign', 'malignant'
                        normal_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            normal_correct += 1
                    if(labels[img_cnt] == 'benign'):    #normal', 'benign', 'malignant'
                        benign_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            benign_correct += 1
                    if(labels[img_cnt] == 'malignant'):    #normal', 'benign', 'malignant'
                        malignant_total += 1
                        if(INbreast_classes[indices[0]] == labels[img_cnt] ):
                            malignant_correct += 1

                    img_cnt += 1
            print('INbreast acc: ',correct/total)
            temp_normal_acc = round(normal_correct / normal_total, 2)
            temp_benign_acc = round(benign_correct / benign_total, 2)
            temp_malignant_acc = round(malignant_correct / malignant_total, 2)
            acc = round(correct/total, 4)
            if(acc> INbreast_stored_acc):
                INbreast_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' in temp_file and 'finetune' in temp_file and 'INbreast'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(temp_normal_acc)}_{str(temp_benign_acc)}_{str(temp_malignant_acc)}_{str(INbreast_stored_acc)}_{model_name}_zeroshoton_INbreast_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")

        correct = 0
        total = 0
        text_inputs = torch.cat([clip.tokenize(f"a figure of {c}") for c in cheXpert_classes]).to(device)
        with torch.no_grad():
            for images, label_tokens,labels in CheXpert5x200_loader:
                images = images.to(device)
                label_tokens = label_tokens.to(device)
                img_cnt = 0
                for image_input in images:
                    image_features = model.encode_image(image_input.unsqueeze(0))
                    text_features = model.encode_text(text_inputs)
                    #CP Network acts on imges features and text features
                    image_features = cp_network(image_features)
                    text_features = cp_network(text_features)
                    # Pick the top 5 most similar labels for the image
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    #take the features in cores
                    image_features = image_features[:,0:cores]
                    text_features = text_features[:,0:cores]

                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    values, indices = similarity[0].topk(len(cheXpert_classes))
                    # Print the result
                    #print("\nTop predictions:\n")
                    #for value, index in zip(values, indices):
                    #    print(f"{classes[index]:>16s}: {100 * value.item():.2f}%")
                    total += 1
                    correct += int((cheXpert_classes[indices[0]] == labels[img_cnt]))
                    img_cnt += 1
            print('CheXpert acc: ',correct/total)
            acc = round(correct/total, 4)
            if(acc> CheXpert_stored_acc):
                CheXpert_stored_acc = acc
                temp_files = os.listdir(opt.model_saved_path)
                #for temp_file in temp_files:
                #    if('FrozenCLIP' in temp_file and 'v2' in temp_file and 'finetune' in temp_file and 'CheXpert5x200'==temp_file.split('_')[-4] \
                #        and 'CP_CLIP' in temp_file and str(nodes)==temp_file.split('_')[-3] and str(cores)==temp_file.split('_')[-2]):
                #        os.remove(opt.model_saved_path+'/'+temp_file)
                checkpoint_path = f"{opt.vision_encoder}_{str(CheXpert_stored_acc)}_{model_name}_zeroshoton_CheXpert5x200_{str(nodes)}_{str(cores)}_finetune"+"v2"+".pth"
                checkpoint = {
                    'it': epoch,
                    'cp_nn': cp_network.state_dict(),
                    #'optimizer': optimizer.state_dict(),
                    #'scheduler': scheduler.state_dict()
                    }
                torch.save(checkpoint, opt.model_saved_path+'/'+checkpoint_path)
                logger.info(f"checkpoint_{epoch} saved")


if __name__ == "__main__":

    if(1):
        print('Zeroshot CLIP')
        ZeroShotCLIP(opt_dataset = 'SIIMACR')
        ZeroShotCLIP(opt_dataset = 'INbreast')
        ZeroShotCLIP(opt_dataset = 'CheXpert5x200')
    
    if(1):
        print('Finetuning CLIP')
        FineTuneCLIP(dataset1 = 'MIMICGAZE', dataset2 = 'INbreast', dataset3 = 'SIIMACR', dataset4 = 'CheXpert5x200')   #finetune on MIMICGAZE, then zero shot on INBreast and SIIMACR

    if(1):
        FineTuneCPCLIP(dataset1 = 'MIMICGAZE', dataset2 = 'INbreast', dataset3 = 'SIIMACR', dataset4 = 'CheXpert5x200')   #finetune on MIMICGAZE, then zero shot on INBreast and SIIMACR
        FineTuneCPCLIP_FrozenCLIP(dataset1 = 'MIMICGAZE', dataset2 = 'INbreast', dataset3 = 'SIIMACR', dataset4 = 'CheXpert5x200')
        FineTuneCPCLIPv2(dataset1 = 'MIMICGAZE', dataset2 = 'INbreast', dataset3 = 'SIIMACR', dataset4 = 'CheXpert5x200')
        FineTuneCPCLIPv2_FrozenCLIP(dataset1 = 'MIMICGAZE', dataset2 = 'INbreast', dataset3 = 'SIIMACR', dataset4 = 'CheXpert5x200')

    if(1):
        FineTuneCPCLIP_GAZE(dataset1 = 'MIMICGAZE', dataset2 = 'INbreast', dataset3 = 'SIIMACR', dataset4 = 'CheXpert5x200')   #finetune on MIMICGAZE, then zero shot on INBreast and SIIMACR
        FineTuneCPCLIP_FrozenCLIP_GAZE(dataset1 = 'MIMICGAZE', dataset2 = 'INbreast', dataset3 = 'SIIMACR', dataset4 = 'CheXpert5x200')
        FineTuneCPCLIPv2_GAZE(dataset1 = 'MIMICGAZE', dataset2 = 'INbreast', dataset3 = 'SIIMACR', dataset4 = 'CheXpert5x200')
        FineTuneCPCLIPv2_FrozenCLIP_GAZE(dataset1 = 'MIMICGAZE', dataset2 = 'INbreast', dataset3 = 'SIIMACR', dataset4 = 'CheXpert5x200')
