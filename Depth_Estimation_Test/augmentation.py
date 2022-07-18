import numpy as np
import skimage.color as color
from skimage import exposure
def image_augmentation(x,contrast,brightness,gamma):
    x=x/255
    x = (0.5 + contrast*(x-0.5)) + brightness
    x = np.minimum(x, 1.0)
    x = np.maximum(x, 0)    
    x= np.power(x,gamma)

    x = np.minimum(x, 1.0)
    x = np.maximum(x, 0)
    x = x/0.5 -1.0
    return x
def image_augmentation_0_1(x,contrast,brightness,gamma):
    x=x/255
    #x=hsv_aug(x,seeds)
    x = (0.5 + contrast*(x-0.5)) + brightness
    x = np.minimum(x, 1.0)
    x = np.maximum(x, 0)    
    x= np.power(x,gamma)

    x = np.minimum(x, 1.0)
    x = np.maximum(x, 0)
    return x

def horizontal_flip(x,depth, random_val):

    if(random_val>0.5):
        x=np.flip(x,1).copy()
        depth=np.flip(depth,1).copy()
    return x,depth


def vertical_flip(x,depth, random_val):
    if(random_val>0.5):
        x=np.flip(x,0).copy()
        depth=np.flip(depth,0).copy()
    return x, depth
def rotate(x,depth,degree):
    x=np.rot90(x,degree,axes=(0,1)).copy()
    depth=np.rot90(depth,degree,axes=(0,1)).copy()
    return x,depth
def randcrop(x,depth,x_seeds,y_seeds,interval_x,interval_y):
    x = x[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x,:]#[H,W,C]
    depth = depth[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x]
    return x ,depth

def horizontal_flip(x,depth, random_val):

    if(random_val>0.5):
        x=np.flip(x,1).copy()
        depth=np.flip(depth,1).copy()
    return x,depth

def vertical_flip(x,depth, random_val):
    if(random_val>0.5):
        x=np.flip(x,0).copy()
        depth=np.flip(depth,0).copy()
    return x, depth
def rotate(x,depth,degree):
    x=np.rot90(x,degree,axes=(0,1)).copy()
    depth=np.rot90(depth,degree,axes=(0,1)).copy()
    return x,depth
def randcrop(x,depth,x_seeds,y_seeds,interval_x,interval_y):
    x = x[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x,:]#[H,W,C]
    depth = depth[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x]
    return x ,depth

# 3d (x: H W C N)

def randcrop_3d(x,depth,x_seeds,y_seeds,interval_x,interval_y):
    x = x[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x,:,:]#[H,W,C,N]
    depth = depth[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x]
    return x ,depth



def randcrop_3d_OF(x,depth,OF,x_seeds,y_seeds,interval_x,interval_y):
    x = x[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x,:,:]#[H,W,C,N]
    depth = depth[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x]
    OF = OF[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x,:,:]#[H,W,C,N]
    return x ,depth, OF

def horizontal_flip_OF(x,depth,OF,random_val):
    if(random_val>0.5):
        x=np.flip(x,1).copy()
        depth=np.flip(depth,1).copy()
        OF = np.flip(OF,1).copy()
    return x ,depth, OF

def vertical_flip_OF(x,depth,OF, random_val):
    if(random_val>0.5):
        x=np.flip(x,0).copy()
        depth=np.flip(depth,0).copy()
        OF = np.flip(OF,0).copy()
    return x ,depth, OF

def rotate_OF(x,depth,OF,degree):
    x=np.rot90(x,degree,axes=(0,1)).copy()
    depth=np.rot90(depth,degree,axes=(0,1)).copy()
    OF = np.rot90(OF,degree,axes=(0,1)).copy()
    return x ,depth, OF

def horizontal_flip_w_conf(x,depth,conf,random_val):
    
    if(random_val>0.5):
        x=np.flip(x,1).copy()
        depth=np.flip(depth,1).copy()
        conf = np.flip(conf,1).copy()
    return x,depth,conf


def vertical_flip_w_conf(x,depth,conf,random_val):
    if(random_val>0.5):
        x=np.flip(x,0).copy()
        depth=np.flip(depth,0).copy()
        conf = np.flip(conf,0).copy()
    return x,depth,conf

def rotate_w_conf(x,depth,conf,degree):
    x=np.rot90(x,degree,axes=(0,1)).copy()
    depth=np.rot90(depth,degree,axes=(0,1)).copy()
    conf = np.rot90(conf,degree,axes=(0,1)).copy()
    return x,depth,conf

def randcrop_3d_w_conf(x,depth,conf,x_seeds,y_seeds,interval_x,interval_y):
    x = x[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x,:,:]#[H,W,C,N]
    depth = depth[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x]
    conf = conf[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x]#[H,W,C,N]
    return x ,depth, conf


#For defocusnet
def horizontal_flip_dfd(x,depth,conf,defocus,random_val):

    if(random_val>0.5):
        x=np.flip(x,1).copy()
        depth=np.flip(depth,1).copy()
        conf = np.flip(conf,1).copy()
        defocus = np.flip(defocus,2).copy()
    return x,depth,conf,defocus


def vertical_flip_dfd(x,depth,conf,defocus,random_val):
    if(random_val>0.5):
        x=np.flip(x,0).copy()
        depth=np.flip(depth,0).copy()
        conf = np.flip(conf,0).copy()
        defocus = np.flip(defocus,1).copy()

    return x,depth,conf,defocus
def rotate_dfd(x,depth,conf,defocus,degree):
    x=np.rot90(x,degree,axes=(0,1)).copy()
    depth=np.rot90(depth,degree,axes=(0,1)).copy()
    conf = np.rot90(conf,degree,axes=(0,1)).copy()
    defocus = np.rot90(defocus,degree,axes=(1,2)).copy()

    return x,depth,conf,defocus
def randcrop_3d_dfd(x,depth,conf,defocus,x_seeds,y_seeds,interval_x,interval_y):
    x = x[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x,:,:]#[H,W,C,N]
    depth = depth[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x]
    conf = conf[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x]#[H,W,C,N]
    defocus = defocus[:,y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x]#[H,W,C,N]

    return x,depth,conf,defocus



def horizontal_flip_dfd_NYU(x,depth,defocus,random_val):
    if(random_val>0.5):
        x=np.flip(x,1).copy()
        depth=np.flip(depth,1).copy()
        defocus = np.flip(defocus,2).copy()
    return x,depth,defocus
def vertical_flip_dfd_NYU(x,depth,defocus,random_val):
    if(random_val>0.5):
        x=np.flip(x,0).copy()
        depth=np.flip(depth,0).copy()
        defocus = np.flip(defocus,0).copy()
    return x,depth,defocus
def rotate_dfd_NYU(x,depth,defocus,degree):
    x=np.rot90(x,degree,axes=(0,1)).copy()
    depth=np.rot90(depth,degree,axes=(0,1)).copy()
    defocus = np.rot90(defocus,degree,axes=(0,1)).copy()
    return x,depth,defocus
def randcrop_3d_dfd_NYU(x,depth,defocus,x_seeds,y_seeds,interval_x,interval_y):
    x = x[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x,:,:]#[H,W,C,N]
    depth = depth[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x]
    defocus = defocus[y_seeds:y_seeds-interval_y,x_seeds:x_seeds-interval_x,:]#[H,W,C,N]
    return x,depth,defocus
