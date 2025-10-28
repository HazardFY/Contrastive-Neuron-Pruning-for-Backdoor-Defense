'''
This is the implement of WaNet [1].

Reference:
[1] WaNet - Imperceptible Warping-based Backdoor Attack. ICLR 2021.
'''

import copy
from copy import deepcopy
import random


import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import pdb
import cupy as cp
import torchvision

from .base import *
from tqdm import tqdm

class AddTrigger:
    def __init__(self):
        pass
    def Fourier_apttern(self,img_,target_img,L,ratio):
        fft_trg_np = np.fft.fft2(target_img, axes=(-2, -1))  # axis int or shape tuple,fft_trg_np:(3,384,384)但是元素是复数
        amp_target, pha_target = np.abs(fft_trg_np), np.angle(fft_trg_np)  # #取绝对值：将复数变化成实数  振幅谱和相位图
        amp_target_shift = np.fft.fftshift(amp_target, axes=(-2, -1))

        fft_source_np = np.fft.fft2(img_, axes=(-2, -1))
        amp_source, pha_source = np.abs(fft_source_np), np.angle(fft_source_np)
        amp_source_shift = np.fft.fftshift(amp_source, axes=(-2, -1))

        # swap the amplitude part of local image with target amplitude spectrum
        c, h, w = img_.shape
        b = (np.floor(np.amin((h, w)) * L)).astype(int)  # 替换块的边长，选择其中最小边然后乘L（0.003）
        # 中心点
        c_h = np.floor(h / 2.0).astype(int)
        c_w = np.floor(w / 2.0).astype(int)

        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1

        

        amp_source_shift[:, h1:h2, w1:w2] = amp_source_shift[:, h1:h2, w1:w2] * (1 - ratio) + (amp_target_shift[:,h1:h2, w1:w2]) * ratio

        amp_source_shift = np.fft.ifftshift(amp_source_shift, axes=(-2, -1))

        # get transformed image via inverse fft
        fft_local_ = amp_source_shift * np.exp(1j * pha_source)
        local_in_trg = np.fft.ifft2(fft_local_, axes=(-2, -1))
        local_in_trg = np.real(local_in_trg)
        return local_in_trg

    def add_trigger(self, img, noise=False):
        """Add WaNet trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).
            noise (bool): turn on noise mode, default is False

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        if noise:
            target_img = random.choice(self.ims_noise)
        else:
            target_img=self.im_target
        # pdb.set_trace()
        poison_img = self.Fourier_apttern(img.numpy(),target_img,self.L,self.mix_ratio)  # CHW
        return torch.from_numpy(poison_img)
    
    



class AddDatasetFolderTrigger(AddTrigger):
    """Add WaNet trigger to DatasetFolder images.

    Args:
        identity_grid (orch.Tensor): the poisoned pattern shape.
        noise_grid (orch.Tensor): the noise pattern.
        noise (bool): turn on noise mode, default is False.
        s (int or float): The strength of the noise grid. Default is 0.5.
        grid_rescale (int or float): Scale :attr:`grid` to avoid pixel values going out of [-1, 1].
            Default is 1.
        noise_rescale (int or float): Scale the random noise from a uniform distribution on the
            interval [0, 1). Default is 2.
    """

    def __init__(self, opt,noise=False):
        super(AddDatasetFolderTrigger, self).__init__()

        
        self.L=opt.L
        self.mix_ratio=opt.mix_ratio
        
        # get the target image and the noise images.
        transforms_list = []
        transforms_list.append(torchvision.transforms.Resize((opt.input_height, opt.input_height)))
        transforms_list.append(torchvision.transforms.ToTensor())  
        transforms_class = Compose(transforms_list)
        
        im_target = Image.open(opt.target_img).convert('RGB')
        im_target = transforms_class(im_target)
        self.im_target = np.clip(im_target.numpy() * 255, 0, 255)# tensor->ndarray
        
        self.ims_noise = []
        noiseImage_list = os.listdir(opt.cross_dir)
        # noiseImage_names = np.random.choice(noiseImage_list,bs)
        for noiseImage_name in noiseImage_list:
            noiseImage_path = os.path.join(opt.cross_dir,noiseImage_name)
            im_noise = Image.open(noiseImage_path).convert('RGB')
            im_noise = transforms_class(im_noise)
            im_noise = np.clip(im_noise.numpy()*255,0,255) # tensor->ndarray
            self.ims_noise.append(im_noise)


        self.noise=noise
        # pdb.set_trace()

    def __call__(self, img):
        """Get the poisoned image.

        Args:
            img (PIL.Image.Image | numpy.ndarray | torch.Tensor): If img is numpy.ndarray or torch.Tensor, the shape should be (H, W, C) or (H, W).
        Returns:
            torch.Tensor: The poisoned image.
        """
        if type(img) == PIL.Image.Image:
            # pdb.set_trace()
            img = F.pil_to_tensor(img)# 先将PIL->tensor [W, H,C]->[C, H, W]，注意，这里处理为tensor是在0-255之间的，添加trigger之后再转化为PIL
            
            img = self.add_trigger(img, noise=self.noise)
            # 1 x H x W
            if img.size(0) == 1:
                img = img.squeeze().numpy()
                img = Image.fromarray(np.clip(img,0,255).round().astype(np.uint8), mode='L')
            # 3 x H x W
            elif img.size(0) == 3:
                img = img.numpy().transpose(1, 2, 0)
                img = Image.fromarray(np.clip(img,0,255).round().astype(np.uint8))
            else:
                raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            # H x W
            if len(img.shape) == 2:
                img = torch.from_numpy(img)
                img = F.convert_image_dtype(img, torch.float)
                img = self.add_trigger(img, noise=self.noise)
                img = img.numpy()

            # H x W x C
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = F.convert_image_dtype(img, torch.float)
                img = self.add_trigger(img, noise=self.noise)
                img = img.permute(1, 2, 0).numpy()

            return img
        elif type(img) == torch.Tensor:
            # H x W
            if img.dim() == 2:
                img = F.convert_image_dtype(img, torch.float)
                img = self.add_trigger(img, noise=self.noise)
            # H x W x C
            else:
                img = F.convert_image_dtype(img, torch.float)
                img = img.permute(2, 0, 1)
                img = self.add_trigger(img, noise=self.noise)
                img = img.permute(1, 2, 0)
            return img
        else:
            raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))




class AddMNISTTrigger(AddTrigger):
    """Add WaNet trigger to MNIST image.

    Args:
        identity_grid (orch.Tensor): the poisoned pattern shape.
        noise_grid (orch.Tensor): the noise pattern.
        noise (bool): turn on noise mode, default is False.
        s (int or float): The strength of the noise grid. Default is 0.5.
        grid_rescale (int or float): Scale :attr:`grid` to avoid pixel values going out of [-1, 1].
            Default is 1.
        noise_rescale (int or float): Scale the random noise from a uniform distribution on the
            interval [0, 1). Default is 2.
    """

    def __init__(self, identity_grid, noise_grid, noise=False, s=0.5, grid_rescale=1, noise_rescale=2):
        super(AddMNISTTrigger, self).__init__()

        self.identity_grid = deepcopy(identity_grid)
        self.noise_grid = deepcopy(noise_grid)
        self.h = self.identity_grid.shape[2]
        self.noise = noise
        self.s = s
        self.grid_rescale = grid_rescale
        grid = self.identity_grid + self.s * self.noise_grid / self.h
        self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        self.noise_rescale = noise_rescale

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = self.add_trigger(img, noise=self.noise)
        # print("img:",img.shape)
        #poison_img = Image.fromarray(np.clip(img*255,0,255).round().astype(np.uint8))
        img = img.squeeze().numpy()
        img = Image.fromarray(np.clip(img*255,0,255).round().astype(np.uint8), mode='L')
        return img


class AddCIFAR10Trigger(AddTrigger):
    """Add WaNet trigger to CIFAR10 image.

    Args:
        identity_grid (orch.Tensor): the poisoned pattern shape.
        noise_grid (orch.Tensor): the noise pattern.
        noise (bool): turn on noise mode, default is False.
        s (int or float): The strength of the noise grid. Default is 0.5.
        grid_rescale (int or float): Scale :attr:`grid` to avoid pixel values going out of [-1, 1].
            Default is 1.
        noise_rescale (int or float): Scale the random noise from a uniform distribution on the
            interval [0, 1). Default is 2.
    """

    def __init__(self, identity_grid, noise_grid, noise=False, s=0.5, grid_rescale=1, noise_rescale=2):
        super(AddCIFAR10Trigger, self).__init__()

        self.identity_grid = deepcopy(identity_grid)
        self.noise_grid = deepcopy(noise_grid)
        self.h = self.identity_grid.shape[2]
        self.noise = noise
        self.s = s
        self.grid_rescale = grid_rescale
        grid = self.identity_grid + self.s * self.noise_grid / self.h
        self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        self.noise_rescale = noise_rescale

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = self.add_trigger(img, noise=self.noise)
        img = img.numpy().transpose(1, 2, 0)
        img = Image.fromarray(np.clip(img*255,0,255).round().astype(np.uint8))
        # img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target


class PoisonedDatasetFolder(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,

                 noise,
                 opt,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedDatasetFolder, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        if poisoned_rate==1.0 and opt.include_ori_targetclass==False:
            samples_temp=list()
            for i in tqdm(range(len(benign_dataset.samples))):
                data = benign_dataset.samples[i]
                if data[1] != opt.target_label:
                    samples_temp.append((data[0],data[1]))
                
            print('target_label',opt.target_label)
            print(len(benign_dataset.samples),'-->',len(samples_temp))
            self.samples=samples_temp
        
        total_num = len(self.samples)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])
     
        # add noise 
        self.noise = noise
        noise_rate = poisoned_rate*2 
        noise_num = int(total_num * noise_rate)
        self.noise_set = frozenset(tmp_list[poisoned_num:poisoned_num+noise_num])# 当poisoned_rate=0.1时，为空集

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([]) # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform) # add noise
        
        
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(  noise=False,opt=opt))
        #add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index, AddDatasetFolderTrigger(  noise=True,opt=opt))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if index in self.poisoned_set:
            sample = self.poisoned_transform(sample)
            target = self.poisoned_target_transform(target)
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            sample = self.poisoned_transform_noise(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            # target = self.poisoned_target_transform(target)

        else:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

        return sample, target


class PoisonedMNIST(MNIST):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 identity_grid, 
                 noise_grid,
                 noise,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedMNIST, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # add noise 
        self.noise = noise
        noise_rate = poisoned_rate * 2
        noise_num = int(total_num * noise_rate)
        self.noise_set = frozenset(tmp_list[poisoned_num:poisoned_num+noise_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([]) # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform) # add noise
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddMNISTTrigger(identity_grid, noise_grid,  noise=False))
        #add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index, AddMNISTTrigger(identity_grid, noise_grid,  noise=True))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            img = self.poisoned_transform_noise(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 identity_grid, 
                 noise_grid,
                 noise,
                 poisoned_transform_index,
                 poisoned_target_transform_index):
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # add noise 
        self.noise = noise
        noise_rate = poisoned_rate * 2
        noise_num = int(total_num * noise_rate)
        self.noise_set = frozenset(tmp_list[poisoned_num:poisoned_num+noise_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.poisoned_transform_noise = Compose([]) # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.poisoned_transform_noise = copy.deepcopy(self.transform) # add noise
        self.poisoned_transform.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger(identity_grid, noise_grid,  noise=False))
        #add noise transform
        self.poisoned_transform_noise.transforms.insert(poisoned_transform_index, AddCIFAR10Trigger(identity_grid, noise_grid,  noise=True))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        # add noise mode
        elif index in self.noise_set and self.noise == True:
            img = self.poisoned_transform_noise(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


def CreatePoisonedDataset(benign_dataset, y_target, poisoned_rate, noise,opt, poisoned_transform_index, poisoned_target_transform_index):
    class_name = type(benign_dataset)
    if class_name == DatasetFolder:
        return PoisonedDatasetFolder(benign_dataset, y_target, poisoned_rate, noise,opt, poisoned_transform_index, poisoned_target_transform_index)
    # elif class_name == MNIST:
    #     return PoisonedMNIST(benign_dataset, y_target, poisoned_rate, noise, poisoned_transform_index, poisoned_target_transform_index)
    # elif class_name == CIFAR10:
    #     return PoisonedCIFAR10(benign_dataset, y_target, poisoned_rate,  noise, poisoned_transform_index, poisoned_target_transform_index)
    else:
        raise NotImplementedError


class FIBA(Base):
    """Construct poisoned datasets with FIBA method.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        target_img (str): the path of target img.
        cross_dir (str): the path of noise dir
        noise(bool): add noise brach or not
        opt:
        poisoned_transform_train_index (int): The position index that poisoned transform will be inserted in train dataset. Default: 0.
        poisoned_transform_test_index (int): The position index that poisoned transform will be inserted in test dataset. Default: 0.
        poisoned_target_transform_index (int): The position that poisoned target transform will be inserted. Default: 0.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 model,
                 loss,
                 y_target,
                 poisoned_rate,
                 
                 noise,
                 opt,
                 poisoned_transform_train_index=0,
                 poisoned_transform_test_index=0,
                 poisoned_target_transform_index=0,
                 schedule=None,
                 seed=0,
                 deterministic=False):
       
        super(FIBA, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        self.poisoned_train_dataset = CreatePoisonedDataset(
            train_dataset,
            y_target,
            poisoned_rate,

            noise,
            opt,
            poisoned_transform_train_index,
            poisoned_target_transform_index)

        self.poisoned_test_dataset = CreatePoisonedDataset(
            test_dataset,
            y_target,
            1.0,
            
            noise,
            opt,
            poisoned_transform_test_index,
            poisoned_target_transform_index)
