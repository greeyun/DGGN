# A Domain Generative Graph Network for EEG-based Emotion Recognition
This is a pytorch implementation of the paper

## Environment
- Pytorch 1.8.1
- Python 3.8.0
- cuda 11.1

## Network structure
![image](https://user-images.githubusercontent.com/68091618/215970027-1d16df5f-28c9-4f54-a8b0-07dd150de709.png)
Fig. 1.  The schematic framework of DGGN.

![image](https://user-images.githubusercontent.com/68091618/215970115-a7008ad7-aa4d-418d-a6c2-a36176e16252.png)
Fig. 2.  The Framework of Generator.

## Dataset
Download target dataset DEAP and SEED from http://www.eecs.qmul.ac.uk/mmv/datasets/deap and https://bcmi.sjtu.edu.cn/~seed/index.html respectively.

## pretrain
'python pretrain_and_gan_dep.py'

## train
'python train_dep.py'
