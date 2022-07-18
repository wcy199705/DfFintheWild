import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2,os
from os import listdir
from os.path import isfile, join
class Real_Scenes(Dataset):
    def __init__(self):
        self.input_size = (352,288)    
        self.root = "Datasets/"
        
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        path = self.root + str(idx) + "/"
        list_dir = os.listdir(path)
        files_dir = [file for file in list_dir if file.endswith('.png') or file.endswith('jpg')]
        files_dir = sorted(files_dir)
        Height,Width,C=cv2.imread(path+ files_dir[0]).shape
        Height = Height
        Width = Width
        FS = np.zeros((Height,Width,3,10),dtype=np.float32)
        focus_dists=[]
        
        with open (path + "focus_distance.txt",'r') as f:
            lines = f.readlines()
            for i in range(0,10):
                line = lines[i]
                focus_distance = float(line)
                focus_dists.append(focus_distance)
                

        with open (path + "focal_length.txt",'r') as f:
            line = f.readline()
            focal_length = float(line)

        
        #reverse = list(reversed(focus_dists))
        focus_dists = np.asarray(focus_dists)
        #need to inverse
        
        relative_Fov = (1/focal_length - 1/focus_dists)
        relative_Fov = relative_Fov / np.min(relative_Fov)
        relative_Fov = np.expand_dims(relative_Fov,axis=0)
        relative_Fov = np.expand_dims(relative_Fov,axis=2)
        relative_Fov = np.expand_dims(relative_Fov,axis=2)
        #relative_Fov = np.expand_dims(relative_Fov,0
        

        focus_dists = np.expand_dims(focus_dists,axis=1)
        focus_dists = np.expand_dims(focus_dists,axis=2)
        focus_dists = 1/focus_dists
        #focus_dists=1/focus_dists
        for i in range(0,10):

            FS[:,:,:,i] = (cv2.imread(path+ files_dir[i] )).astype(np.float32)
        FS=FS/127.5 -1.0
        FS = torch.from_numpy(np.transpose(FS,(2,3,0,1)))
        if Height % 32 != 0:
            pad_h = 32 - (Height % 32)
        else:
            pad_h = 0 
        if Width % 32 != 0:
            pad_w = 32 - (Width % 32)
        else:
            pad_w =0

        FS  = np.pad(FS,
                    ( (0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                    mode='constant',
                    constant_values=(-1, -1)
        )
        return FS, torch.Tensor(focus_dists), torch.Tensor(relative_Fov)
