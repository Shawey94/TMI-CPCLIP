# CPCLIP

# [Core-Periphery Multi-Modality Feature Alignment for Zero-Shot Medical Image Analysis](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10721320)

In this work, we applied the core-periphery principle derived from functional brain networks for multi-modality feature alignment in CLIP.

<p align="left"> 
<img width="800" src="https://github.com/Shawey94/TMI-CPCLIP/blob/main/CPCLIP.png">
</p>

### Environment (Python 3.8.12)
```
# Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh

# Install required packages
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```


### Datasets:

- ChestXray
- CheXpert5x200
- INbreast
- MIMIC-GAZE
- SIIMACR
- TMED2

### Generating CP Graphs:
Please adjust the total number of nodes and core nodes as needed.
```
python CP_graph_generator.py
```

### Training:
```
python CP_CLIPFineTune.py --lr_clip 0.000001 --epochs 50 --batch_size 16 --lr_cpNN 0.001 --gpu_id 0
```

### Visualization:
After the model is well-trained, please update the model path and image path in GradCAM_CP_CLIP.py to generate attention maps (visualization).
```
python GradCAM_CP_CLIP.py
```


### Citation:
```
@article{Yu2024CPCLIP,
  title={Core-Periphery Multi-Modality Feature Alignment for Zero-Shot Medical Image Analysis},
  author={Yu, Xiaowei and Zhang, Lu and Wu, Zihao and Dajiang Zhu},
  journal={IEEE Transactions on Medical Imaging},
  year={2024}
}

@article{Yu2024GyriSulci,
  title={Gyri vs. sulci: Core-periphery organization in functional brain networks},
  author={Yu, Xiaowei and Zhang, Lu and Cao, Chao and Chen, Tong and Lyu, Yanjun and Zhang, Jing and Liu, Tianming and Dajiang Zhu},
  journal={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2024}
}

@article{Yu2024CPMiccai,
  title={Cp-clip: Core-periphery feature alignment clip for zero-shot medical image analysis},
  author={Yu, Xiaowei and Wu, Zihao and Zhang, Lu and Zhang, Jing and Lyu, Yanjun and Dajiang Zhu},
  journal={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2024}
}
```
