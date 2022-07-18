import numpy as np
import torch
import torch.nn.functional as F
import random
from scipy import  io
from torch.utils.data import Dataset
import cv2,h5py,os,math, time
from os import listdir
from os.path import isfile, join
from augmentation import *
import OpenEXR, Imath
from tqdm import tqdm

def DDFF_vertical_flip(x,depth, random_val):
    if(random_val>0.5):
        x=np.flip(x,1).copy()
        depth=np.flip(depth,0).copy()
    return x, depth
def DDFF_rotate(x,depth,degree):
    x=np.rot90(x,degree,axes=(1,2)).copy()
    depth=np.rot90(depth,degree,axes=(0,1)).copy()
    return x,depth
def DDFF_horizontal_flip(x,depth, random_val):
    
    if(random_val>0.5):
        x=np.flip(x,2).copy()
        depth=np.flip(depth,1).copy()
    return x,depth


class FocalStackDDFFH5Reader(Dataset):
    
    def __init__(self, hdf5_filename, stack_key="stack_train", disp_key="disp_train"):
        self.hdf5 = h5py.File(hdf5_filename, 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        focal_length = 521.4052
        K2 = 1982.0250823695178
        flens = 7317.020641763665
        baseline = K2 / flens * 1e-3
        focus_dists = np.linspace(baseline * focal_length / 0.5,baseline * focal_length / 7,num=10)
        focus_dists = np.expand_dims(focus_dists,axis=1)
        focus_dists = np.expand_dims(focus_dists,axis=2)

        self.focus_dists = torch.Tensor(np.tile(focus_dists,[1,224,224]))
        self.min_dist = np.min(focus_dists)
        self.max_dist = np.max(focus_dists)

        self.focus_dists = (self.focus_dists-self.min_dist)/(self.max_dist - self.min_dist)
    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]

    def __getitem__(self, idx):
        #Create sample dict
        contrast,brightness,gamma,flip_x,flip_y, angle=self.get_seeds()
        FS=self.hdf5[self.stack_key][idx].astype(np.float32)
        gt=self.hdf5[self.disp_key][idx].astype(np.float32)
        if self.stack_key == "stack_train":
            FS=image_augmentation(FS,contrast,brightness,gamma)
            
            FS,gt = DDFF_horizontal_flip(FS,gt,flip_x)
            FS,gt = DDFF_vertical_flip(FS,gt,flip_y)
            FS,gt = DDFF_rotate(FS,gt,angle)
        elif self.stack_key == "stack_val":
            FS = FS/127.5 -1.0
        mask=torch.from_numpy(np.where(gt==0.0,0.,1.).astype(np.bool_))

        gt = (gt - self.min_dist)  / (self.max_dist - self.min_dist)
        
        FS= torch.from_numpy(np.transpose(FS,(3,0,1,2)))

        gt = torch.from_numpy(gt)
        H,W=(224,224)
        
        return FS, gt , self.focus_dists, mask

    def get_stack_size(self):
        return self.__getitem__(0)['input'].shape[0]
    def get_seeds(self):
        return (random.uniform(0.4,1.6),random.uniform(-0.1,0.1),random.uniform(0.5,2.0),random.uniform(0,1.0),random.uniform(0,1.0),random.randint(0,3))
class FS6_dataset(Dataset):
    def __init__(self,mode):
        self.root = "Datasets/fs_6/" + mode +"/"
        self.imglist_all = [f for f in listdir(self.root) if isfile(join(self.root, f)) and f[-7:] == "All.tif"]
        self.imglist_dpt = [f for f in listdir(self.root) if isfile(join(self.root, f)) and f[-7:] == "Dpt.exr"]
        self.imglist_all.sort()
        self.imglist_dpt.sort()
        self.max_depth = 3.0
        focus_dists = np.array([0.1,0.15,0.3,0.7,1.5])
        focus_dists = np.expand_dims(focus_dists,axis=1)
        focus_dists = np.expand_dims(focus_dists,axis=2).astype(np.float32)
        self.mode=mode
        self.Focus_Dists = torch.Tensor(np.tile(focus_dists,[1,256,256]))
    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, index):
        img_dpt= self.read_dpt (self.root + self.imglist_dpt[index]) 
        
        contrast,brightness,gamma,flip_x,flip_y,angle=self.get_seeds()
        img_index = index * 5
        mats_input = np.zeros((256, 256, 3, 0))
        for i in range(5):
            img = cv2.imread(self.root + self.imglist_all[img_index + i])
            mats_input = np.concatenate((mats_input,np.expand_dims(img,axis=-1)), axis=3)
        
        if self.mode=="train":
            mats_input=image_augmentation(mats_input,contrast,brightness,gamma)
            mats_input,img_dpt = horizontal_flip(mats_input,img_dpt,flip_x)
            mats_input,img_dpt = vertical_flip(mats_input,img_dpt,flip_y)
            mats_input,img_dpt = rotate(mats_input,img_dpt,angle)
            img_dpt[img_dpt< 0.0] = 0.0
            img_dpt[img_dpt > 2.0] = 0.0

        elif self.mode == "test":
            mats_input = mats_input/127.5 -1.0
            img_dpt[img_dpt< 0.1] = 0.0
            img_dpt[img_dpt > 1.5] = 0.0

        mats_input = np.transpose(mats_input,(2,3,0,1))
        
        mask=torch.from_numpy(np.where(img_dpt==0.0,0.,1.).astype(np.bool_))

        
        img_dpt= torch.Tensor(img_dpt)
        mats_input=torch.Tensor(mats_input)

        return mats_input,\
            img_dpt,\
            self.Focus_Dists,\
            mask
    def read_dpt(self,img_dpt_path):
        dpt_img = OpenEXR.InputFile(img_dpt_path)
        dw = dpt_img.header()['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        (r, g, b) = dpt_img.channels("RGB")
        dpt = np.fromstring(r, dtype=np.float16)
        dpt.shape = (size[1], size[0])
        return dpt
    def get_seeds(self):
        return (random.uniform(0.4,1.6),random.uniform(-0.1,0.1),random.uniform(0.5,2.0),random.uniform(0,1.0),random.uniform(0,1.0),random.randint(0,3))

class FlyingThings3d(Dataset):
    def __init__(self,mode):
        self.mode = mode
        self.train_size=(256,256)
        self.num_imgs=15
        self.input_size=(540,960)
        self.cropping = (self.input_size[0] - self.train_size[0],self.input_size[1] - self.train_size[1])
        self.rgb_paths = [[] for i in range(self.num_imgs)]
        self.disp_paths = []
        self.low_bound = 10
        self.high_bound = 100
        self.focus_dists = np.linspace(self.low_bound,self.high_bound,self.num_imgs)
        self.focus_dists = np.expand_dims(self.focus_dists,axis=1)
        self.focus_dists = np.expand_dims(self.focus_dists,axis=2).astype(np.float32)
        with open ("Datasets/FlyingThings3D_FS/"+ mode + "/flyingthings3d_FS_path.txt",'r') as f:
            for line in tqdm(f.readlines(),desc="flyingthings"):
                tmp = line.strip().split()
                for i in range(self.num_imgs):
                    self.rgb_paths[i].append(tmp[i])
                self.disp_paths.append(tmp[-1])
        
    def __len__(self):
        return len(self.disp_paths)

    def __getitem__(self,idx):#TEST/Train
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        depth = cv2.imread(self.disp_paths[idx],cv2.IMREAD_UNCHANGED) #depth check range, shape

        imgs= np.concatenate([np.expand_dims(cv2.imread(x[idx]),axis=3) for x in self.rgb_paths],3) #C*H*W
        if self.mode=="train":
            y_crop,x_crop,contrast,brightness,gamma,flip_x,flip_y,angle=self.get_seeds()
            imgs,depth = randcrop_3d(imgs,depth,x_crop,y_crop,self.cropping[1],self.cropping[0])
            imgs=image_augmentation(imgs,contrast,brightness,gamma)
            imgs,depth = horizontal_flip(imgs,depth,flip_x)
            imgs,depth = vertical_flip(imgs,depth,flip_y)
            imgs,depth = rotate(imgs,depth,angle)    
            Focus_Dists = torch.Tensor(np.tile(self.focus_dists,[1,self.train_size[0],self.train_size[1]]))

            imgs=torch.Tensor(np.transpose(imgs,(2,3,0,1)))
        elif self.mode == "val":
            imgs = imgs/127.5 -1.0
            imgs=torch.Tensor(np.transpose(imgs,(2,3,0,1)))
            C,N,H,W = imgs.shape        

            if H % 32 != 0:
                pad_h = 32 - (H % 32)
            else:
                pad_h = 0 
            if W % 32 != 0:
                pad_w = 32 - (W % 32)
            else:
                pad_w =0

            imgs  = np.pad(imgs,
                        ( (0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                        mode='constant',
                        constant_values=(-1, -1)
            )
            Focus_Dists = torch.Tensor(np.tile(self.focus_dists,[1,self.input_size[0]+pad_h,self.input_size[1]+pad_w]))

            #pad_h = 32 -(self.input_size[0]%32) if self.input_size[0] >0 else 0

        depth[depth< 0.0] = 0.0
        mask=torch.from_numpy(np.where(depth==0.0,0.,1.).astype(np.bool_))
        depth= torch.Tensor(depth)

        return imgs,\
            depth,\
            mask,\
            Focus_Dists

    def get_seeds(self):
        return (random.randint(0,self.cropping[0]-1),random.randint(0,self.cropping[1]-1),random.uniform(0.4,1.6),random.uniform(-0.1,0.1),random.uniform(0.5,2.0),random.uniform(0,1.0),random.uniform(0,1.0),random.randint(0,3))
class HCI_dataset(Dataset):
    
    def __init__(self, hdf5_filename, stack_key="stack_train", disp_key="disp_train"):
        self.hdf5 = h5py.File(hdf5_filename, 'r')
        self.stack_key = stack_key
        self.disp_key = disp_key
        self.input_size = (512,512)
        if stack_key == "stack_train":
            self.size = (256,256)

        elif stack_key == "stack_val":
            self.size = (512,512)

        self.cropping = (self.input_size[0] - self.size[0],self.input_size[1] - self.size[1])

        focus_dists = self.hdf5['focus_position_disp']
        focus_dists = np.squeeze(focus_dists,axis=0)
        focus_dists = np.expand_dims(focus_dists,axis=1)
        focus_dists = np.expand_dims(focus_dists,axis=2)
        
        self.focus_dists = torch.Tensor(np.tile(focus_dists,[1,self.size[0],self.size[1]]))
        self.min_dist = np.min(focus_dists)
        self.max_dist = np.max(focus_dists)
    def __len__(self):
        return self.hdf5[self.stack_key].shape[0]

    def __getitem__(self, idx):
        FS=self.hdf5[self.stack_key][idx].astype(np.float32)
        FS_re = np.zeros((512,512,3,10),dtype=np.float32)
        for i in range(0,10):
            FS_re[:,:,:,i] = FS[i,:,:,:]
        gt=self.hdf5[self.disp_key][idx].astype(np.float32)
        if self.stack_key == "stack_train":
            y_crop,x_crop,contrast,brightness,gamma,flip_x,flip_y,angle=self.get_seeds()
            FS,gt = randcrop_3d(FS_re,gt,x_crop,y_crop,self.cropping[1],self.cropping[0])
            FS=image_augmentation(FS,contrast,brightness,gamma)
            FS,gt = horizontal_flip(FS,gt,flip_x)
            FS,gt = vertical_flip(FS,gt,flip_y)
            FS,gt = rotate(FS,gt,angle)
        elif self.stack_key == "stack_val":
            FS=FS_re/127.5 -1.0
            gt[gt< self.min_dist] = -3.0
            gt[gt > self.max_dist] = -3.0
        
        mask=torch.from_numpy(np.where(gt==-3.0,0.,1.).astype(np.bool_))
        FS = torch.from_numpy(FS.transpose((2,3,0,1)))
        gt = torch.from_numpy(gt)
        return FS, gt , self.focus_dists, mask

    def get_stack_size(self):
        return self.__getitem__(0)['input'].shape[0]
    def get_seeds(self):
        return (random.randint(0,self.cropping[0]-1),random.randint(0,self.cropping[1]-1),random.uniform(0.4,1.6),random.uniform(-0.1,0.1),random.uniform(0.5,2.0),random.uniform(0,1.0),random.uniform(0,1.0),random.randint(0,3))
class Smartphone(Dataset):
    
    def __init__(self, mode, num_imgs):
        self.mode = mode
        self.num_imgs = num_imgs
        self.input_size = (504,378)
        self.center_crop = (336,252)
        self.rand_crop = (224,224)
        self.cropping = (self.center_crop[0] - self.rand_crop[0],self.center_crop[1] - self.rand_crop[1])
        self.indexes = np.rint(np.linspace(0,48,num_imgs,endpoint=True)).astype(np.int)
        self.focus_dists = []
        #https://storage.googleapis.com/cvpr2020-af-data/LearnAF%20Dataset%20Readme.pdf
        focus_dists = [3910.92,2289.27,1508.71,1185.83,935.91,801.09,700.37,605.39,546.23,486.87,447.99,407.40,379.91,350.41,329.95,307.54,
                            291.72,274.13,261.53,247.35,237.08,225.41,216.88,207.10,198.18,191.60,183.96,178.29,171.69,165.57,160.99,155.61,150.59,146.81,
                            142.35,138.98,134.99,131.23,127.69,124.99,121.77,118.73,116.40,113.63,110.99,108.47,106.54,104.23,102.01]
        for index in self.indexes:
            self.focus_dists.append(focus_dists[index])
        self.focus_dists = np.expand_dims(self.focus_dists,axis=1)
        self.focus_dists = np.expand_dims(self.focus_dists,axis=2).astype(np.float32)
        self.focus_dists = self.focus_dists*0.001
        self.Fovs = (1/0.00444)-(1/np.array(self.focus_dists))#https://www.devicespecifications.com/en/model/121b4c25
        self.Fovs = self.Fovs/np.min(self.Fovs)
        self.Fovs = np.expand_dims(self.Fovs,axis=0)
        if mode == "train":
            self.focus_dists = torch.Tensor(np.tile(self.focus_dists,[1,self.rand_crop[0],self.rand_crop[1]]))
        elif mode == "test":
            self.focus_dists = torch.Tensor(np.tile(self.focus_dists,[1,self.center_crop[0]+16,self.center_crop[1]+4]))
        self.focus_dists=1/self.focus_dists
        self.max_depth = 1/0.10201  
        self.min_depth = 1/3.91092
        self.root= 'Datasets/Real_data_DP/'
        self.depths=[]
        self.confids=[]
        self.FS=[]
        if mode == "train":
            for i in tqdm(range(1,8),desc="trainset"):
                path = self.root + mode + str(i) + '/'
                scenes=os.listdir(path+'scaled_images/')
                for scene in scenes:
                    self.depths.append(path + 'merged_depth/'+ scene +'/' + 'result_merged_depth_center.png')
                    self.confids.append(path + 'merged_conf/'+ scene +'/' + 'result_merged_conf_center.exr')
                    FS_imgs=[]
                    for j in self.indexes:
                        FS_imgs.append(path + 'scaled_images/'+ scene +'/' + str(j)+ '/result_scaled_image_center.jpg')
                    self.FS.append(FS_imgs)
        elif mode == "test":
            path = self.root + mode  + '/'
            scenes=os.listdir(path+'scaled_images/')
            for scene in scenes:
                self.depths.append(path + 'merged_depth/'+ scene +'/' + 'result_merged_depth_center.png')
                self.confids.append(path + 'merged_conf/'+ scene +'/' + 'result_merged_conf_center.exr')
                FS_imgs=[]
                for j in self.indexes:
                    FS_imgs.append(path + 'scaled_images/'+ scene +'/' + str(j)+ '/result_scaled_image_center.jpg')
                self.FS.append(FS_imgs)
            
        
    def __len__(self):
        return len(self.depths)

    def __getitem__(self, idx):
        #Create sample dict
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        
        #base_focal_length = self.FS_focal_length[idx][48]
        FS = np.zeros((self.center_crop[0],self.center_crop[1],self.num_imgs,3),dtype=np.float32)
        for i in range(0,self.num_imgs):
            img = cv2.imread(self.FS[idx][i]).astype(np.float32)[:,:,:]
            FS[:,:,i,:] = img[84:-84,63:-63,:].astype(np.float32)
        img = cv2.imread(self.FS[idx][self.num_imgs-1]).astype(np.float32)[:,:,:]
        FS[:,:,(self.num_imgs-1),:] = img[84:-84,63:-63,:].astype(np.float32)

        gt =cv2.imread(self.depths[idx],cv2.IMREAD_UNCHANGED).astype(np.float32)[84:-84,63:-63]
        gt = gt/255.0
        gt = (20)/(100-(100-0.2)*gt)
        gt=1/gt
        conf = cv2.imread(self.confids[idx],cv2.IMREAD_UNCHANGED )[84:-84,63:-63,-1]
        conf [conf>1.0] = 1.0

        if self.mode == "train":
            y_crop,x_crop,contrast,brightness,gamma,flip_x,flip_y,angle=self.get_seeds()
            FS,gt,conf = randcrop_3d_w_conf(FS,gt,conf,x_crop,y_crop,self.cropping[1],self.cropping[0])
            FS=image_augmentation(FS,contrast,brightness,gamma)
            FS,gt,conf = horizontal_flip_w_conf(FS,gt,conf,flip_x)
            FS,gt,conf = vertical_flip_w_conf(FS,gt,conf,flip_y)
            FS,gt,conf = rotate_w_conf(FS,gt,conf,angle)
            
        elif self.mode == "test":
            FS=FS/127.5 -1.0
        gt[gt< self.min_depth] = 0.0
        gt[gt > self.max_depth] = 0.0
        
        mask=torch.from_numpy(np.where(gt==0.0,0.,1.).astype(np.bool_))

        FS = torch.from_numpy(np.transpose(FS,(3,2,0,1)))
        N,C,H,W = FS.shape
        if H % 32 != 0:
            pad_h = 32 - (H % 32)
        else:
            pad_h = 0 
        if W % 32 != 0:
            pad_w = 32 - (W % 32)
        else:
            pad_w =0
        FS = F.pad(torch.Tensor(FS),(0,pad_w,0,pad_h))#top 4 padding

        gt = torch.from_numpy(gt)
        return FS, gt , self.focus_dists, mask, conf, torch.from_numpy(self.Fovs)

    def get_seeds(self):
        return (random.randint(0,self.cropping[0]-1),random.randint(0,self.cropping[1]-1),random.uniform(0.4,1.6),random.uniform(-0.1,0.1),random.uniform(0.5,2.0),random.uniform(0,1.0),random.uniform(0,1.0),random.randint(0,3))
