# Predicting-Event-Memorability-from-Contextual-Visual-Semantics
This repository contains pytorch implementation of five configurations in our paper "Predicting Event Memorability from Contextual Visual Semantics".
1. Raw images are to be put in '../datasets/r3/images/'
2. Train and validation (val) splits for different configurations are under '../datasets/r3/splits/'; the set of train_1.txt, val_1.txt, etc. contains image names and memorability scores for the respective split.
3. Configurations of ablation study are with individual folders, e.g., './no_face', './no_activity', etc. './full_set' is for full configuration without removing features. 
4. Complete extrinsic features and the memory test outcome is available in 'R3_data.csv' file. Description of the features is given in 'R3_data_notes.txt'. Both can be download together with the original image cues @ https://drive.google.com/drive/folders/1Bx_ePv7ui6DCIXkESCpoyuvd0H3B9o6d?usp=sharing

########################################################################################

To train AMNet and CEMNet_wt_AMNet:

python3 main.py --train-batch-size 128 --test-batch-size 128 --cnn ResNet50FC --dataset lamem --train-split train_1 --val-split val_1

To predict:

python3 main.py --cnn ResNet50FC --model-weights /path/to/model/weights_xx.pkl --eval-images /path/to/evl_images --csv-out memorabilities.txt

To train other models (ICNet, MLP, CEMNet_wt_ICNet):

[Go the the respective folder, e.g., '../ICNet']

python main.py

To predict (please select corresponding splits and model in predict.py):

python predict.py

[Where necessary, change Dataset.py to the corresponding directory of split]

########################################################################################

System configuration:

platform: UBUNTU 16.04

GPU: GeForce GTX 1080

CUDA:9.0

########################################################################################

Python packages:

python 3.5.6

pytorch 0.2.0

Torchvison 0.1.9

Numpy 1.15.2

Opencv 3.1.0

PIL 6.1.0

########################################################################################

To cite the paper: 
Xu Q., Fang F., del Molino A.G, Subbaraju V., Lim J.H., Predicting Event Memorability from Contextual Visual Semantics, NeurIPS 2021.

If you have any questions, please feel free to contact Dr Xu Qianli: qxu@i2r.a-star.edu.sg
