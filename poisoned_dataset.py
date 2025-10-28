import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import pdb
import torchvision
import PIL.Image as Image
from copy import deepcopy

import models
import config
import core
def img_read(img_path):
    return Image.open(img_path).convert('RGB')
def get_dataset(trainset,testset,args):

    dataset = torchvision.datasets.DatasetFolder
    if args.attack_method=='badnet':
        pattern = torch.zeros((3, args.input_height, args.input_height), dtype=torch.uint8)
        # 设置右下角的元素为黑色
        # pattern[0, -3:, -3:] = 255
        length=args.patch_size
        for i in range(1,length+1):# [1,length+1)
            for j in range(1,length+1):
                if (i+j)%2 == 0:
                    pattern[:,-i,-j] = 255
                else:
                    pattern[:,-i,-j] = 0
        # trigger对应的权重，在badnet中一般设置为1，如果是blended可能就是0.1等等
        weight = torch.zeros((3, args.input_height, args.input_height), dtype=torch.float32)
        weight[:, -length:, -length:] = 1.0
        
        badnets = core.BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.ResNet(18),
            # model=core.models.BaselineMNISTNetwork(),
            loss=torch.nn.CrossEntropyLoss(),
            y_target=args.target_label,
            poisoned_rate=args.poisoned_rate,
            pattern=pattern,
            weight=weight,
            opt=args,
            poisoned_transform_train_index=args.poisoned_transform_train_index,
            poisoned_transform_test_index=args.poisoned_transform_test_index,
            poisoned_target_transform_index=args.poisoned_target_transform_index,
            schedule=None,
            seed=args.seed
        )
        
        poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()
    elif args.attack_method=='Blended':
        blended_ratio=args.mix_ratio
        target_img=img_read('/media/userdisk1/yf/CVPR2023/BackdoorBox-main/samples/HelloKitty.png')

        target_trans=transforms.Compose([
            transforms.Resize((args.input_height,args.input_height)),
            transforms.ToTensor()
        ])
        pattern=target_trans(target_img)# 注意这里pattern需要0-255，而ToTensor会将图像归一化
        pattern=pattern*255

        weight=torch.ones(3,args.input_height,args.input_height)
        weight=weight*blended_ratio


        blendednets = core.Blended(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.ResNet(18),
            # model=core.models.BaselineMNISTNetwork(),
            loss=torch.nn.CrossEntropyLoss(),
            y_target=args.target_label,
            poisoned_rate=args.poisoned_rate,
            pattern=pattern,
            weight=weight,
            opt=args,
            poisoned_transform_train_index=args.poisoned_transform_train_index,
            poisoned_transform_test_index=args.poisoned_transform_test_index,
            poisoned_target_transform_index=args.poisoned_target_transform_index,
            schedule=None,
            seed=args.seed
        )
        poisoned_train_dataset, poisoned_test_dataset = blendednets.get_poisoned_dataset()
    elif args.attack_method=='WaNet':
        work_dir,_=os.path.split(args.test_model)
        identity_grid=torch.load(os.path.join(work_dir,'identity_grid.pth'))
        noise_grid=torch.load(os.path.join(work_dir,'noise_grid.pth'))

        WaNet = core.WaNet(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.ResNet(18),
            # model=core.models.BaselineMNISTNetwork(),
            loss=torch.nn.CrossEntropyLoss(),
            y_target=args.target_label,
            poisoned_rate=args.poisoned_rate,
            identity_grid=identity_grid,
            noise_grid=noise_grid,
            noise=True, #确定是否添加noise brench
            opt=args,
            poisoned_transform_train_index=args.poisoned_transform_train_index,
            poisoned_transform_test_index=args.poisoned_transform_test_index,
            poisoned_target_transform_index=args.poisoned_target_transform_index,
            schedule=None,
            seed=args.seed,
            
        )

        poisoned_train_dataset, poisoned_test_dataset = WaNet.get_poisoned_dataset()
    elif args.attack_method=='FIBA':
        FIBA = core.FIBA(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.ResNet(18),
            # model=core.models.BaselineMNISTNetwork(),
            loss=torch.nn.CrossEntropyLoss(),
            y_target=args.target_label,
            poisoned_rate=args.poisoned_rate,
            noise=True, #确定是否添加noise brench
            opt=args,
            poisoned_transform_train_index=args.poisoned_transform_train_index,
            poisoned_transform_test_index=args.poisoned_transform_test_index,
            poisoned_target_transform_index=args.poisoned_target_transform_index,
            schedule=None,
            seed=args.seed
            
        )

        poisoned_train_dataset, poisoned_test_dataset = FIBA.get_poisoned_dataset()
    else:
        raise Exception("Wrong attack method")
    
    return poisoned_test_dataset