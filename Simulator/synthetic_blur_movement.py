#Fast bilateral-space stereo for synthetic defocus (john barron)
#Code Reference: https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel

import os, errno,random
from re import X
import cv2
import numpy as np
import scipy.io as io
import argparse
import time
from tqdm import tqdm
import mat73
import torch
import torch.nn as nn
def FOV_warp(x, Fov,beta,gamma): #Fov was affected by alpha value which is an error of focus distance

    x = x.astype(np.float32)
    x = torch.from_numpy(np.rollaxis(x,2,0))
    x = torch.unsqueeze(x,0)
    B,C,H,W = x.shape

    xx = torch.unsqueeze(torch.linspace(-1, 1,steps=(W)),dim=0).repeat(H,1).view(1,1,H,W).repeat(1,1,1,1)
    yy = torch.unsqueeze(torch.linspace(-1, 1,steps=(H)),dim=1).repeat(1,W).view(1,1,H,W).repeat(1,1,1,1)
    grid2 = torch.cat((xx,yy),1).float().to(x.device)
    grid2[:,0,:,:] = ((W//2)*(( Fov -1 ) * grid2[:,0,:,:].clone())) - beta
    grid2[:,1,:,:] = ((H//2 )*((Fov -1 ) * grid2[:,1,:,:].clone())) - gamma

    #flow unit: pixel
    xx = torch.unsqueeze(torch.arange(0, W),dim=0).repeat(H,1).view(1,1,H,W).repeat(1,1,1,1)
    yy = torch.unsqueeze(torch.arange(0, H),dim=1).repeat(1,W).view(1,1,H,W).repeat(1,1,1,1)

    grid = torch.cat((xx,yy),1).float().to(x.device)
    #flow unit: pixel
    grid = torch.autograd.Variable(grid) - grid2.clone()
    grid[:,0,:,:] = 2.0*grid[:,0,:,:].clone() / max(W-1,1)-1.0
    grid[:,1,:,:] = 2.0*grid[:,1,:,:].clone() / max(H-1,1)-1.0
    grid = grid.permute(0,2,3,1).type(torch.float32)#permute(0,2,3,1)   
    output = nn.functional.grid_sample(x, grid,align_corners=True)
    output = np.transpose(np.squeeze(output.numpy()),(1,2,0))

    return output
def DepthFOV_warp(x, Fov,beta,gamma): 
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x,0)
    x = torch.unsqueeze(x,0)

    B,C,H,W = x.shape

    xx = torch.unsqueeze(torch.linspace(-1, 1,steps=(W)),dim=0).repeat(H,1).view(1,1,H,W).repeat(1,1,1,1)
    yy = torch.unsqueeze(torch.linspace(-1, 1,steps=(H)),dim=1).repeat(1,W).view(1,1,H,W).repeat(1,1,1,1)
    grid2 = torch.cat((xx,yy),1).float().to(x.device)
    
    grid2[:,0,:,:] = ((W//2)*(( Fov -1 ) * grid2[:,0,:,:].clone())) - beta
    grid2[:,1,:,:] = ((H//2 )*((Fov -1 ) * grid2[:,1,:,:].clone())) - gamma

    #flow unit: pixel
    xx = torch.unsqueeze(torch.arange(0, W),dim=0).repeat(H,1).view(1,1,H,W).repeat(1,1,1,1) 
    yy = torch.unsqueeze(torch.arange(0, H),dim=1).repeat(1,W).view(1,1,H,W).repeat(1,1,1,1)

    grid = torch.cat((xx,yy),1).float().to(x.device)
    #flow unit: pixel
    grid = torch.autograd.Variable(grid) - grid2.clone()
    grid[:,0,:,:] = 2.0*grid[:,0,:,:].clone() / max(W-1,1)-1.0
    grid[:,1,:,:] = 2.0*grid[:,1,:,:].clone() / max(H-1,1)-1.0
    grid = grid.permute(0,2,3,1).type(torch.float32)#permute(0,2,3,1)   
    output = nn.functional.grid_sample(x, grid,align_corners=True)
    output = np.squeeze(output.numpy())


    return output
#Code Reference: https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel
def check_dir(path_):
    '''Check directory if exist, if not, create directory using given path'''
    if not os.path.exists(path_):
        try:
            os.makedirs(path_)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
def create_blur(blur_size):
    circ_size=np.zeros([blur_size,blur_size])
    center_offset=(blur_size//2,blur_size//2)

    """create a disk of the given radius and center"""
    c_kernel=cv2.circle(circ_size, center_offset, blur_size//2, (1,1,1), -1)
    return c_kernel/np.sum(c_kernel)




parser = argparse.ArgumentParser(description='Synthetic dataset with scene movements')
parser.add_argument('--dataset',default='NYU_move_out_0_1/',type=str, help='Dataset')
parser.add_argument('--focal_length',default=0.028,type=float, help='focal length in meter')
parser.add_argument('--F_num',default=2.0,type=float, help='f number of lens')
parser.add_argument('--pixel_vs_meter',default=1/(0.0000014)*352/4080,type=float, help='pixel/meter (correpond to sensor size)')
parser.add_argument('--num_imgs',default=10,type=int, help='number of images to use in focal stacks')
parser.add_argument('--num_planes',default=2000,type=int, help='number of kinds of planes')
parser.add_argument('--max_depth',default=1.0,type=float, help='Maximum of depth ranges')
parser.add_argument('--min_depth',default=0.1,type=float, help='Minimum of depth ranges')

args = parser.parse_args()

if args.dataset == 'NYU_move_out_0_1/':
    
    height =224 
    width = 352        
    num_planes =args.num_planes
    save_dir = args.dataset
    NYU_matfile=mat73.loadmat("nyu_depth_v2_labeled.mat")
    Images=NYU_matfile['images']
    Images = Images [16:-16,16:-16,:,:]
    Depths=NYU_matfile['depths'][16:-16,16:-16,:]
    Depths=Depths.astype(np.float64)
    N = args.num_imgs
    Num_imgs=1449
    start = time.time()


    for img_idx in tqdm(range(0,Num_imgs),desc="Making",mininterval=1):
        random_choice_devices = random.randint(0,3)
        if random_choice_devices==0:
            #pixel4_XL
            size_ratio = width/4032
            alpha_slope = -0.00266
            y_intercept = 0.019155
            beta_mean = -4.45515
            beta_var = 7.18485
            gamma_mean = -9.9504701
            gamma_var = 8.04556863
            focal_length = 0.0044
            F_num = 1.7
        elif random_choice_devices==1:
            #pixel6
            size_ratio = width/4080
            alpha_slope = -0.00429249

            y_intercept = 0.00330253

            beta_mean = 0.470281
            beta_var = 6.2634662
            gamma_mean = 2.69174424
            gamma_var = 6.859772247
            focal_length = 0.0068
            F_num = 1.9

        elif random_choice_devices==2:
            #galaxy_S8+
            size_ratio = width/4032
            alpha_slope = -0.00203839
            y_intercept =0.0166955
            beta_mean = 4.430173117
            beta_var = 4.60067699
            gamma_mean = 3.695449964
            gamma_var = 3.589144555
            focal_length = 0.0043
            F_num = 1.5
        elif random_choice_devices==3:
            #Galaxy_note10
            size_ratio = width/4032
            alpha_slope = -0.00402384
            y_intercept = 0.0247385
            beta_mean = -4.315575939
            beta_var = 2.9198626
            gamma_mean = -0.9456601
            gamma_var = 0.153538997
            focal_length = 0.0048
            F_num = 1.7
        save_path = save_dir + str(img_idx) +'/'
        check_dir(save_dir+str(img_idx))
        depth = Depths[:,:,img_idx]
        depth = cv2.resize(depth,(width,height))
        depth = args.max_depth*(depth - np.min(depth))/(np.max(depth) - np.min(depth))
        depth = depth +args.min_depth


        origin_depth = depth.copy()
        focal_length = focal_length * args.pixel_vs_meter
        lens_dia = focal_length/F_num
        max_scene_depth = np.max(depth)
        min_scene_depth = np.min(depth)
        depth_pixel = depth * args.pixel_vs_meter

        min_focus_dist = 0.1
        max_focus_dist = 0.9
        focus_dists= (1/np.linspace(1/max_focus_dist,1/min_focus_dist,N,endpoint=True)) #uniform
        
        disparity=np.zeros((height,width,N),dtype=np.float64)        
        max_focus_dist_pixel = max_focus_dist*args.pixel_vs_meter 
        min_focus_dist_pixel = min_focus_dist*args.pixel_vs_meter 

        min_AFOV = 1/(focal_length*min_focus_dist_pixel/(min_focus_dist_pixel-focal_length))
        max_AFOV = 1/(focal_length*max_focus_dist_pixel/(max_focus_dist_pixel-focal_length)) 
        origin_max_AFOV = max_AFOV/min_AFOV + alpha_slope*(1/max_scene_depth) + y_intercept
        camera_setting = {"focal_length": focal_length,
                "aperture_size": lens_dia,
                "pixel_mm": args.pixel_vs_meter,
                "max_focus_dist" : max_scene_depth,
                "min_focus_dist" : min_scene_depth
                }#IT WILL BE SAVED as a mat file
        
        for num in range(N):#in pixel values
            image = Images[:,:,:,img_idx]
            image = image.astype(np.float32)
            image = cv2.resize(image,(width,height))
            focus_dist = focus_dists[num]
            focus_dist_pixel = args.pixel_vs_meter*focus_dist
            lens_to_sensor=focal_length*focus_dist_pixel/(focus_dist_pixel-focal_length)
            if num!=0:
                Fov = (1/lens_to_sensor)
                alpha = alpha_slope*(1/focus_dist) + y_intercept

                origin_Fov = Fov/min_AFOV + alpha
                FoV = origin_max_AFOV/origin_Fov
                #get beta, gamma values
                beta = random.normalvariate(beta_mean,beta_var) * size_ratio
                gamma = random.normalvariate(gamma_mean,gamma_var) * size_ratio
                beta = torch.from_numpy(np.asarray(beta))
                gamma = torch.from_numpy(np.asarray(gamma))
                image =  FOV_warp(image,FoV,beta,gamma)

                depth_pixel_now = DepthFOV_warp(depth_pixel,FoV,beta,gamma)
            else:
                depth_pixel_now = depth_pixel
            coc_scale=lens_to_sensor*lens_dia/focus_dist_pixel
            coc_min_max_dis=[]
            ind_count=0
            
            disparity[:,:,num] = np.abs(np.squeeze(coc_scale*(depth_pixel_now - focus_dist_pixel)/depth_pixel_now))
            for k in range(args.num_planes):
                min_dis=k/num_planes*(max_scene_depth-min_scene_depth) +min_scene_depth
                max_dis=(k+1)/num_planes*(max_scene_depth-min_scene_depth) +min_scene_depth
                sub_dis=(min_dis+(max_dis-min_dis)/2)
                coc_size=round(coc_scale*(sub_dis-focus_dist)/sub_dis)
                if k > 0:
                    if max_dis==max_scene_depth: #last iteration
                        max_dis+=0.1
                    if coc_min_max_dis[ind_count-1][0] == coc_size:
                        coc_min_max_dis[ind_count-1][2] = max_dis
                    else:
                        coc_min_max_dis.append([int(coc_size),min_dis,max_dis])
                        ind_count+=1
                else:
                    coc_min_max_dis.append([int(coc_size),min_dis,max_dis])
                    ind_count+=1
                num_coc_layers=len(coc_min_max_dis)
                '''list to keep sub-image and sub-depth'''
                blurred_imgs=[]
                depth_set=[]
            for i in range(num_coc_layers):
                coc_size=coc_min_max_dis[i][0]
                min_dis=coc_min_max_dis[i][1]
                max_dis=coc_min_max_dis[i][2]
                                
                sub_depth=(np.where((depth >= min_dis) & (depth < max_dis), 1, 0)).astype(np.uint8)#???
                '''subimage based on depth layer, weighted by matting_ratio'''
                sub_img=(image).astype(np.uint8)
                depth_set.append(sub_depth)

                if coc_size == 0:
                    coc_size=1                
                kernel = create_blur(2*abs(coc_size)+1)
                '''combined final output image'''
                blurred_img= cv2.filter2D(sub_img,-1,kernel)
                blurred_img=cv2.cvtColor(blurred_img,cv2.COLOR_BGR2RGB)
                blurred_imgs.append(blurred_img)    
                
            blurred_img=blurred_imgs[num_coc_layers-1]*np.expand_dims(depth_set[num_coc_layers-1],axis=2)
            for i in range(num_coc_layers-1):
                blurred_img+=blurred_imgs[num_coc_layers-2-i]*np.expand_dims(depth_set[num_coc_layers-2-i],axis=2)
            cv2.imwrite(save_path+"img"+str(num)+".png",blurred_img)
        origin_depth = DepthFOV_warp(origin_depth,FoV,beta,gamma)#assume last one has smallest Fov.
        depth_dict={"depth":origin_depth,"defocus":disparity}
        if np.min(origin_depth)==0:
            exit()
        io.savemat(save_path+"depth.mat",depth_dict)
        io.savemat(save_path+"camera_param.mat",camera_setting)
        exit()
print("avg_time: ",time.time() - start/Num_imgs)

        
