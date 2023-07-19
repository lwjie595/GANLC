from ops import *
import torchvision
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader

import cv2 as cv
import collections, os, math
import numpy as np
from scipy import signal
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import cv2 as cv
from torchvision import utils as vutils
from scipy.io import loadmat
import glob
import  random
class inference_dataset(Dataset):
    def __init__(self, args):
        root = args.input_video_dir
        mode = 'test'
        self.transform = transforms.Compose(
            [transforms.Resize((args.crop_size1, args.crop_size2)), transforms.ToTensor()])
        self.aligned = True
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/Sinogram' % mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, '%s/FBP' % mode) + '/*.*'))

        self.files_B = sorted(glob.glob(os.path.join(root, '%s/Phantom' % mode) + '/*.*'))

    def __getitem__(self, index):
        len_for_each = [224, 320, 128, 234, 239, 210, 343, 214, 244, 211]
        start_for_next = [224, 544, 672, 906, 1145, 1355, 1698, 1912, 2156, 2366]
        Sinogram, Fbp, Phantom = [], [], []
        for i in range(9):
            if (index < start_for_next[i]) & (index > start_for_next[i] - 2):
                index = start_for_next[i] - 2
                break
        print(self.files_A[(index )])
        sinogram = self.transform(Image.fromarray(np.load(self.files_A[(index ) % len(self.files_A)])))
        fbp = self.transform(Image.fromarray(np.load(self.files_C[(index ) % len(self.files_C)])))
        phantom = self.transform(Image.fromarray(np.load(self.files_B[(index ) % len(self.files_B)])))
        Sinogram.append(sinogram.unsqueeze(0))
        Fbp.append(fbp.unsqueeze(0))
        Phantom.append(phantom.unsqueeze(0))
        Sinogram = torch.cat(Sinogram, dim=0)
        Fbp = torch.cat(Fbp, dim=0)
        Phantom = torch.cat(Phantom, dim=0)
        # return {'fbp': fbp, 'phantom': phantom, 'sinogram': sinogram}
        return [Fbp.float(), Phantom.float(), Sinogram.float()]

    def __len__(self):
        return max([len(self.files_A), len(self.files_B), len(self.files_C)])


class train_dataset(Dataset):


    def __init__(self,args):
        root=args.input_video_dir
        mode=args.mode
        self.rnn_num=args.RNN_N
        if mode=='inference':
            mode='test'
        self.transform = transforms.Compose(
            [transforms.Resize((args.crop_size1, args.crop_size2)), transforms.ToTensor()])
        self.aligned=True
        self.files_A = sorted(glob.glob(os.path.join(root, '%s/Sinogram' % mode) + '/*.*'))
        self.files_C = sorted(glob.glob(os.path.join(root, '%s/FBP' % mode) + '/*.*'))

        self.files_B = sorted(glob.glob(os.path.join(root, '%s/Phantom' % mode) + '/*.*'))


    def __getitem__(self, index):
        start_for_next = [224, 544, 778, 1017, 1227, 1570, 1784, 2028, 2238]
        if (index==126):
            a=1
        Sinogram,Fbp,Phantom=[],[],[]
        for i in range(9):
            if (index<start_for_next[i]) &(index>start_for_next[i]-2):
                index=start_for_next[i]-2
                break
        for i in range(self.rnn_num):

            sinogram = self.transform(Image.fromarray(np.load(self.files_A[(index+i) % len(self.files_A)])))
            fbp = self.transform(Image.fromarray(np.load(self.files_C[(index+i) % len(self.files_C)])))
            phantom = self.transform(Image.fromarray(np.load(self.files_B[(index+i) % len(self.files_B)])))
            Sinogram.append(sinogram.unsqueeze(0))
            Fbp.append(fbp.unsqueeze(0))
            Phantom.append(phantom.unsqueeze(0))
        Sinogram = torch.cat(Sinogram, dim=0)
        Fbp = torch.cat(Fbp, dim=0)
        Phantom = torch.cat(Phantom, dim=0)
        # return {'fbp': fbp, 'phantom': phantom, 'sinogram': sinogram}
        p=Phantom.cpu().detach().numpy().squeeze()
        return [ Fbp.float(),Phantom.float(),Sinogram.float()]

    def __len__(self):
        return max([len(self.files_A), len(self.files_B), len(self.files_C)])

