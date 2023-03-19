# Mixup-Inference-in-Training

This is the implementation of our CVPR'23 paper [On the Pitfall of Mixup for Uncertainty Calibration](https://dengbaowang.github.io/). In the paper, we conduct a series of empirical studies showing the calibration issue of Mixup, and propose a new mixup training strategy to address this issue.

## Dependencies
This code requires the following:

* Python 3.6, 
* numpy 1.22.3, 
* Pytorch 1.8.1+cu111, 
* torchvision 0.9.1+cu111.

## Training
For example, you can:

1. Download CIFAR-10 dataset into `./data/`.

2. Run the following demos:
```
python main.py  --dataset cifar10 --arch resnet18 --method ce --seed 101

python main.py  --dataset cifar10 --arch resnet18 --method mixup --alpha 1.0 --seed 101

python main.py  --dataset cifar10 --arch resnet18 --method MIT-L --alpha 1.0 --margin 0 --seed 101

python main.py  --dataset cifar10 --arch resnet18 --method MIT-A --alpha 1.0 --margin 0 --seed 101

python main.py  --dataset cifar10 --arch resnet18 --method MIT-A --alpha 1.0 --margin 0.5 --seed 101

```

## Citation
```
@inproceedings{CVPR23Wang,
author = {Deng-Bao Wang, Lanqing Li, Peilin Zhao, Pheng-Ann Heng, Min-Ling Zhang},
title = {On the Pitfall of Mixup for Uncertainty Calibration},
booktitle = {Proceedings of the 34th IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year = {2023}
}
```

## Contact
If you have any further questions, please feel free to send an e-mail to: wangdb@seu.edu.cn.
