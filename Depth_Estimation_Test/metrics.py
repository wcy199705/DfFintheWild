import numpy as np
import skimage.filters as skf
import torch
def abs_rel(est_depth,gt_depth):
    out = np.abs(gt_depth-est_depth)/(gt_depth)
    total_pixels=np.count_nonzero(~np.isinf(out))
    out[np.isinf(out)]=0
    return np.sum(out)/total_pixels

def sq_rel(est_depth,gt_depth):
    out =np.power((gt_depth-est_depth),2)/gt_depth
    total_pixels=np.count_nonzero(~np.isinf(out))
    out[np.isinf(out)]=0
    return np.sum(out)/total_pixels
def mae(est_depth,gt_depth):
    return np.mean(np.abs(gt_depth-est_depth))

def mse(est_depth,gt_depth):
    return np.mean(np.power((gt_depth-est_depth),2))
def rmse(est_depth,gt_depth):
    return np.sqrt(mse(est_depth,gt_depth))

def rmse_log(est_depth,gt_depth):
    gt_depth = np.log(gt_depth)
    est_depth = np.log(est_depth)
    total_pixels=np.count_nonzero((~np.isinf(est_depth))*(~np.isinf(gt_depth)))
    out=np.power((gt_depth - est_depth),2)
    out[np.isinf(out)]=0

    return np.sqrt(np.sum(out)/total_pixels)
    #Dp=np.nan_to_num(Dp)

def accuracy_k(est_depth,gt_depth,k):
    thresh = np.maximum(est_depth/gt_depth , gt_depth/est_depth)
    total_pixels=np.count_nonzero(~np.isinf(thresh ))
    #Dp=np.nan_to_num(Dp)
    Dp=np.where(thresh <(1.25**k),1,0)
    return (np.sum(Dp))/total_pixels

#https://github.com/albert100121/AiFDepthNet/blob/master/test.py
def get_bumpiness(gt, algo_result, mask, clip=0.05, factor=100):
    # init
    if type(gt) == torch.Tensor:
        gt = gt.cpu().numpy()[0, 0]
    if type(algo_result) == torch.Tensor:
        algo_result = algo_result.cpu().numpy()[0, 0]
    if type(mask) == torch.Tensor:
        mask = mask.cpu().numpy()[0, 0]
    # Frobenius norm of the Hesse matrix
    diff = np.asarray(algo_result - gt, dtype='float64')
    dx = skf.scharr_v(diff)
    dy = skf.scharr_h(diff)
    dxx = skf.scharr_v(dx)
    dxy = skf.scharr_h(dx)
    dyy = skf.scharr_h(dy)
    dyx = skf.scharr_v(dy)
    bumpiness = np.sqrt(
        np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
    bumpiness = np.clip(bumpiness, 0, clip)
    # return bumpiness
    return np.mean(bumpiness[mask]) * factor

def get_bumpiness_non_mask(gt, algo_result, clip=0.05, factor=100):
    # init
    if type(gt) == torch.Tensor:
        gt = gt.cpu().numpy()[0, 0]
    if type(algo_result) == torch.Tensor:
        algo_result = algo_result.cpu().numpy()[0, 0]
    # Frobenius norm of the Hesse matrix
    diff = np.asarray(algo_result - gt, dtype='float64')
    dx = skf.scharr_v(diff)
    dy = skf.scharr_h(diff)
    dxx = skf.scharr_v(dx)
    dxy = skf.scharr_h(dx)
    dyy = skf.scharr_h(dy)
    dyx = skf.scharr_v(dy)
    bumpiness = np.sqrt(
        np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
    bumpiness = np.clip(bumpiness, 0, clip)
    # return bumpiness
    return np.mean(bumpiness) * factor



def AIF_DepthNEt_abs_rel(est,gt,mask):
    return np.mean(np.abs(est[mask] - gt[mask])/gt[mask])
def AIF_DepthNEt_sq_rel(est,gt,mask):
    return np.mean(((est[mask] - gt[mask])**2)/gt[mask])

def mask_abs_rel(est_depth,gt_depth,mask):
    return  np.mean(np.abs(gt_depth[mask]-est_depth[mask])/(gt_depth[mask]))

def mask_sq_rel(est_depth,gt_depth,mask):
    return np.mean(np.power((gt_depth[mask]-est_depth[mask]),2)/(gt_depth[mask]))

def mask_mse(est_depth,gt_depth,mask):
    return np.mean(np.power((gt_depth[mask]-est_depth[mask]),2))

def mask_mae(est_depth,gt_depth,mask):
    return np.mean(np.abs(gt_depth[mask]-est_depth[mask]))

def mask_rmse(est_depth,gt_depth,mask):
    return np.sqrt(np.mean(np.power(est_depth[mask]-gt_depth[mask],2)))

def mask_rmse_log(est_depth,gt_depth,mask):
    gt_depth = np.log(gt_depth[mask])
    est_depth = np.log(est_depth[mask])
    out=np.power((gt_depth - est_depth),2)
    return np.sqrt(np.mean(out))
    #Dp=np.nan_to_num(Dp)

def mask_accuracy_k(est_depth,gt_depth,k,mask):
    A=est_depth[mask]/gt_depth[mask]
    B=gt_depth[mask]/est_depth[mask]
    #A=np.nan_to_num(A)
    #B=np.nan_to_num(B)
    thresh = np.maximum(A , B)
    total_pixels=np.sum(mask)
    #Dp=np.nan_to_num(Dp)
    Dp=np.where(thresh <(1.25**k),1,0)
    return (np.sum(Dp))/total_pixels

def mask_mse_w_conf(est_depth,gt_depth,conf,mask):
    return np.sum(conf[mask]*(np.power((gt_depth[mask]-est_depth[mask]),2)))/np.sum(conf[mask])

def mask_mae_w_conf(est_depth,gt_depth,conf,mask):
    return np.sum(conf[mask]*(np.abs(gt_depth[mask]-est_depth[mask])))/np.sum(conf[mask])

def mask_mse_w_conf_wo_mask(est_depth,gt_depth,conf,mask):
    return np.sum(conf*(np.power((gt_depth-est_depth),2)))/np.sum(conf)

def mask_mae_w_conf_wo_mask(est_depth,gt_depth,conf,mask):
    return np.sum(conf*(np.abs(gt_depth-est_depth)))/np.sum(conf)

