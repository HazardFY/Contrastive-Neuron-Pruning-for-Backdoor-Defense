from logging import raiseExceptions
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

from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import cv2

import models
import config
import core
import poisoned_dataset



args=config.get_arguments().parse_args()

if os.path.exists(args.output_dir):
    print('dir exists',args.output_dir)
    args.mask_file=os.path.join(args.output_dir,args.mask_file)
    print(args.mask_file)
else:
    raise Exception("no dir")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def img_read(img_path):
    return Image.open(img_path).convert('RGB')

def main():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),

    ])
    
    # load model checkpoints and trigger
    if args.dataset=='cifar10':
        net = getattr(models, args.arch)(num_classes=10)
    else:
        raise Exception("Wrong dataset")
    net.load_state_dict(torch.load(args.test_model, map_location=device))
    net = net.to(device)
    # reate poisoned / clean test set
    dataset = torchvision.datasets.DatasetFolder

    trainset = dataset(
        root=os.path.join(args.data_root,'train'),# './data/cifar10/cifar10/train'
        loader=img_read,
        extensions=('png',),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None)
    testset = dataset(
        root=os.path.join(args.data_root,'test'),
        loader=img_read,
        extensions=('png',),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None)
    poisoned_test_dataset=poisoned_dataset.get_dataset(trainset,testset,args)
    poison_test=poisoned_test_dataset
    clean_test=testset
    poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Step 3: pruning
    mask_values = read_data(args.mask_file)
    mask_values = sorted(mask_values, key=lambda x: float(x[2]))
    print('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    cl_loss, cl_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
    po_loss, po_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
    print('0 \t None     \t None     \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))

    results = evaluate_by_number(
        net, mask_values, pruning_max=args.pruning_max, pruning_step=args.pruning_step,
        criterion=criterion, clean_loader=clean_test_loader, poison_loader=poison_test_loader
    )
    file_name = os.path.join(args.output_dir, 'pruning_by_{}.txt'.format(args.pruning_by))
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
        f.writelines(results)


def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values


def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)


def evaluate_by_number(model, mask_values, pruning_max, pruning_step, criterion, clean_loader, poison_loader):
    results = []
    length=len(mask_values)
    nb_max=int(length*pruning_max)  # maximum number of neurons to prune
    nb_step=int(length*pruning_step) # step size
    for start in range(0, nb_max + 1, nb_step):
        i = start
        for i in range(start, start + nb_step):
            pruning(model, mask_values[i])
        layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
        cl_loss, cl_acc = test(model=model, criterion=criterion, data_loader=clean_loader)
        po_loss, po_acc = test(model=model, criterion=criterion, data_loader=poison_loader)
        print('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
        results.append('{} \t {} \t {} \t {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
    return results




def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


if __name__ == '__main__':
    main()

