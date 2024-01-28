# Toward 3D Face Reconstruction in Perspective Projection: Estimating 6DoF Face Pose From Monocular Image
   
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/pdf/2205.04126v2.pdf)

This repository contains the official PyTorch implementation of:

**Toward 3D Face Reconstruction in Perspective Projection: Estimating 6DoF Face Pose From Monocular Image**   

![](assets/perspnet.pdf) 

## Installation
  
 **Setup python environment**
 
   * **CUDA>=10.1** 
   * Python = 3.7
   * PyTorch >= 1.7.1 
 
 **Install other dependencies**
 
   pip install -r requirement.txt

**Install libs**

  1. Please follow  `./lib/mesh/README.md`.
  
  2. Please follow `./lib/mesh_p/README.md`.
  
  3. Please follow `./lib/Sim3DR/README.md`.

## Demo

   Download our pretraind model from [here](https://drive.google.com/file/d/1K9rAmQ7Hduz1on9SWnyLHf_ZxfLwlTOy/view?usp=drive_link) and put it in the folder `./checkpoint/run1/latest_net_R.pth`. 

```sh
$ python demo.py
```
  
## Download Dataset

** Download ARKitFace dataset**

   Please contact email cbsropenproject@outlook.com. We will send you an agreement for signing, and after our verification, the download link for ARKitFace dataset will be sent to you. Please ensure compliance with agreement and do not use this dataset for any non research purposes. For more information about the dataset, please refer to this [here](https://github.com/o0Helloworld0o-ustc/ECCV2022_WCPA_track2).
   
   Then put the download dataset in the folder `./dataset/ARKitFace`. 

** Download BIWI dataset**
   
  Please download the BIWI dataset from https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database and transform GT pose to RGB image, and make a biwi_test.csv file based on that of ARKitFace dataset and the code of./data/biwi_preprocess.py.  


## Testing and training

   Download our pretraind model from [here](https://drive.google.com/file/d/1K9rAmQ7Hduz1on9SWnyLHf_ZxfLwlTOy/view?usp=drive_link) and put it in the folder `./checkpoint/run1/latest_net_R.pth`. 
    
**Testing for ARKitFace dataset**  

```sh
$ python -u test.py --csv_path_test test.csv
```


**Testing for BIWI dataset**

```sh
$ python -u test.py --csv_path_test biwi_test.csv --dataset_mode biwi 
```


**Training**  

```sh
$ python -u train.py  --csv_path_train train.csv --csv_path_test test.csv
```

## Citation

If you find our work usefull in your research, please use the following BibTex entry.

```latex
@ARTICLE{10127617,
  author={Kao, Yueying and Pan, Bowen and Xu, Miao and Lyu, Jiangjing and Zhu, Xiangyu and Chang, Yuanzhang and Li, Xiaobo and Lei, Zhen},
  journal={IEEE Transactions on Image Processing}, 
  title={Toward 3D Face Reconstruction in Perspective Projection: Estimating 6DoF Face Pose From Monocular Image}, 
  year={2023},
  volume={32},
  number={},
  pages={3080-3091},
  doi={10.1109/TIP.2023.3275535}}
```
 


 
 
  
