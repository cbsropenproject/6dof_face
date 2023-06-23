# Toward 3D Face Reconstruction in Perspective Projection: Estimating 6DoF Face Pose From Monocular Image
   
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/pdf/2205.04126v2.pdf)

This repository contains the official PyTorch implementation of:

**Toward 3D Face Reconstruction in Perspective Projection: Estimating 6DoF Face Pose From Monocular Image**   

![](assets/perspnet.png) 

## Installation
  
 **Setup python environment**
 
   * **CUDA>=10.1** 
   * Python = 3.7
   * PyTorch >= 1.7.1 
 
 **install other dependencies**
   pip install -r requirement.txt

**Install libs**

  1. Please follow  `./lib/mesh/README.md`.
  
  2. Please follow `./lib/mesh_p/README.md`.
  
  3. Please follow `./lib/Sim3DR/README.md`.

## Demo

```sh
$ python demo.py
```
  
## Download Dataset

 coming soon

## Testing

**Download our pretraind model from (link coming soon)**

​	Put this model in the folder `./checkpoint/run1/latest_net_R.pth`. 
    
**Testing**  

```sh
$ python -u test.py --csv_path_test test.csv
```

## Citation

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
 


 
 
  