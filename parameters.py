import torch
import argparse
import os

nodes_lookup_table = {'RN50': 1024, 'RN101': 512, 'ViT-B/16': 512, 'ViT-B/32': 512}

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0', help='gpu ids')
parser.add_argument('--num_patches', type=int, default=196, help='number of patches') # (224/16) * (224/16)
parser.add_argument('--epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--weight_decay', default=1e-2, type=float)
parser.add_argument('--opt', default='adamw', type=str)
parser.add_argument('--lr_clip', default=0.000001, type=float)  #1e-6 , 1e-3 for siimacr, 1e-6, 1e-4 for inbreast, chexpert
parser.add_argument('--lr_cpNN', default=0.001, type=float)
parser.add_argument('--lr_cpNN_fzCLIP', default=0.01, type=float)
parser.add_argument('--warmup', default=10, type=int)
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay')
parser.add_argument('--model_saved_path', type=str, default='./models_saved')
parser.add_argument('--log_step', type=int, default=10, help='log_step')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--warm_up', default=8, type=int)
parser.add_argument('--cp_graph_path', type=str, default='./CPGraphs')
parser.add_argument('--INbreast_path', type=str, default='./data/INbreast.txt', help='the directory of INbreast dataset')
parser.add_argument('--SIIMACR_path', type=str, default='./data/SIIMACR.txt', help='the directory of SIIM-ACR dataset')
parser.add_argument('--CheXpert5x200_path', type=str, default='./data/chexpert5x200.txt', help='the directory of CheXpert dataset')
parser.add_argument('--ChestXray_path', type=str, default='./data/chestxray.txt', help='the directory of ChestXray dataset')
parser.add_argument('--MIMICGAZE_path', type=str, default='./data/MIMICGAZE_CLIP_train.txt', help='the directory of MIMIC GAZE dataset')
parser.add_argument('--imagenet_path', type=str, default='/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/CP_ViT_ImageNet/ImageNet1K', help='the directory of imagenet dataset')
parser.add_argument('--TinyImagenet_path', type=str, default='/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/TinyImageNet', help='the directory of tiny imagenet dataset')
parser.add_argument('--ImagenetV2', type=str, default='/media/shawey/cf54ec8b-5d7c-4924-b13e-4ece5630451c/DatasetsZoo/ImageNetV2-matched-frequency', help='the directory of imagenetV2 dataset')
parser.add_argument('--TMED2_path', type=str, default='./data/TMED2.txt', help='the directory of TMED2 dataset')
parser.add_argument('--ratio', type=float, default=0.8, help='the cores number, 256 is 50 percent of the total')
parser.add_argument('--vision_encoder', type=str, default='ViT-B/16', help='RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16')
opt = parser.parse_args()


print("Torch version:", torch.__version__)
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if cuda else 'cpu')

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
