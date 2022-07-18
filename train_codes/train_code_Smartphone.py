import numpy as np
import torch.nn as nn
import torch
import time
import gc
from Depth_Estimation_Network import Network
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import *
import argparse

from train_Dataloader import Smartphone



def masked_MSE_loss(est,gt,conf,mask):
    out = torch.sum(conf[mask]*(torch.pow((est[mask]-gt[mask]),2)))/torch.sum(conf[mask])
    return out 


def main():
    parser = argparse.ArgumentParser(description='Train code: Depth from focus')
    parser.add_argument('--saveroot',default="train_test/",type=str,help='save root')
    parser.add_argument('--lr',type=float,help='learning rate')
    parser.add_argument('--max_epoch',default=1000,type=int,help='max epoch')
    parser.add_argument('--load_epoch',default=0,type=int,help='load epoch')
    parser.add_argument('--batch_size',default=4,type=int,help='batch size')
    parser.add_argument('--cpus',default=4,type=int,help='num_workers')
    args = parser.parse_args()
    root = args.saveroot
    print('root is',root)
    writer = SummaryWriter(log_dir=root+'logs')
    mid_weight=0.3
    load_epoch= args.load_epoch
    Weight1=0.5
    Weight2=0.7
    Weight3=1.0
    batch_size=args.batch_size
    max_epoch=args.max_epoch
    test_epoch=1 
    save_epoch=1
    min_depth = 1/3.91092
    max_depth = 1/0.10201
    num_train=355
    num_val=47
    Height = 336
    Width = 252
    avg_Loss=0.0
    avg_middle=0.0
    avg_DFF_1=0.0
    avg_DFF_2=0.0
    avg_DFF_3=0.0
    avg_photo_loss=0.0
    start=time.time()
    model=Network()
    model=model.cpu()

    train_dataset=Smartphone('train',10)
    valid_dataset=Smartphone('test',10)

    model = nn.DataParallel(model)
    if(load_epoch>1):
        path=root+'models/'+str(load_epoch)+'.pth'
        model.module.load_state_dict(torch.load(path))
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.99))#hyperparmeters from AAAI2020 lr 0.00001 , betas=(0,0.9) #70 epoch 0.001->0.0001
    model=model.cuda()
    dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=args.cpus,pin_memory=True)
    valid_dataloader=DataLoader(valid_dataset,1,shuffle=False,num_workers=args.cpus,pin_memory=True)
    #amp
    for epoch in range(load_epoch,max_epoch+1):#chang validation part
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        if(epoch%save_epoch==0 and epoch!=load_epoch):
            path=root+'models/' + str(epoch)+'.pth'
            torch.save(model.module.state_dict(),path)
        if(epoch%test_epoch==0 and epoch !=load_epoch):
            model.eval()
            with torch.no_grad():
                Avg_mse=0.0
                Avg_mae=0.0

                val_time=0.0
                for idx, samples in enumerate(tqdm(valid_dataloader,desc="valid")):
                    valid_input, test_gt_depth , test_focus_dists, test_mask, test_conf, _= samples
                    test_gt_depth=test_gt_depth.numpy()
                    
                    test_mask = np.squeeze(test_mask.data.cpu().numpy())
                    test_conf = np.squeeze(test_conf.data.cpu().numpy())

                    test_gt_depth = np.squeeze(test_gt_depth)
                    test_focus_dists=test_focus_dists.cuda()   
                    start= time.time()            
                    _,_, _, test_pred3 = model(valid_input,test_focus_dists)
                    val_time = val_time+ (time.time() -start)

                    test_pred3=test_pred3.data.cpu().numpy()#[0,29]
                    test_pred3 = test_pred3[0,:Height,:Width]

                    Avg_mse = Avg_mse + mask_mse_w_conf(test_pred3,test_gt_depth,test_conf,test_mask)
                    Avg_mae = Avg_mae + mask_mae_w_conf(test_pred3,test_gt_depth,test_conf,test_mask)

                print("Avg_mse(" +str(epoch)+") : " ,Avg_mse/num_val)
                print("Avg_mae(" +str(epoch)+") : " ,Avg_mae/num_val)
                print("AVG_time:",val_time/num_val)
                writer.add_scalar("Loss/validation/DFF/Avg_mse",Avg_mse/num_val,epoch)
                writer.add_scalar("Loss/validation/DFF/Avg_mae",Avg_mae/num_val,epoch)
        model.train()
        for idx, samples in enumerate(tqdm(dataloader,desc="Train")): #check variable ranges, images
            train_input, train_gt_depth , train_focus_dists, train_mask, train_conf,_ = samples

            train_input=train_input.cuda(non_blocking=True)
            train_gt_depth=train_gt_depth.cuda(non_blocking=True)
            train_focus_dists=train_focus_dists.cuda(non_blocking=True)
            train_mask=train_mask.cuda(non_blocking=True)
            train_conf=train_conf.cuda(non_blocking=True)
            mid_out,pred1, pred2, pred3=model(train_input,train_focus_dists)
            mid_out = mid_out[:,:,:]
            pred1 = pred1[:,:,:]
            pred2 = pred2[:,:,:]
            pred3 = pred3[:,:,:]

            optimizer.zero_grad()
            Loss1 =masked_MSE_loss((pred1-min_depth)/(max_depth-min_depth),(train_gt_depth-min_depth)/(max_depth-min_depth),train_conf,train_mask)#,gt_gradient,gt_sobel)
            Loss2 =masked_MSE_loss((pred2-min_depth)/(max_depth-min_depth),(train_gt_depth-min_depth)/(max_depth-min_depth),train_conf,train_mask)#,gt_gradient,gt_sobel)
            Loss3 =masked_MSE_loss((pred3-min_depth)/(max_depth-min_depth),(train_gt_depth-min_depth)/(max_depth-min_depth),train_conf,train_mask)#,gt_gradient,gt_sobel)

            mid_loss = masked_MSE_loss((mid_out-min_depth)/(max_depth-min_depth),(train_gt_depth-min_depth)/(max_depth-min_depth),train_conf,train_mask)
            
            Total_Loss = (Weight1*Loss1) + (Weight2*Loss2) + (Weight3* Loss3) + (mid_weight*mid_loss)
            Total_Loss = Total_Loss
            Total_Loss.backward()
            optimizer.step()

            avg_Loss=avg_Loss+Total_Loss.detach().data
            avg_middle=avg_middle+mid_loss.detach().data

            avg_DFF_1=avg_DFF_1+Loss1.detach().data
            avg_DFF_2=avg_DFF_2+Loss2.detach().data
            avg_DFF_3=avg_DFF_3+Loss3.detach().data
            
            
        print("Epoch:",epoch,"AVG_DFF_TotalLoss:",avg_Loss/(num_train))
        writer.add_scalar("Loss/train/Total loss",avg_Loss/num_train,epoch)
        writer.add_scalar("Loss/train/Mid loss",avg_middle/num_train,epoch)
        writer.add_scalar("Loss/train/First/L1 loss",avg_DFF_1/num_train,epoch)
        writer.add_scalar("Loss/train/Second/L1 loss",avg_DFF_2/num_train,epoch)
        writer.add_scalar("Loss/train/Third/L1 loss",avg_DFF_3/num_train,epoch)
        writer.add_scalar("Loss/train/OF loss",avg_photo_loss/num_train,epoch)

        start=time.time()
        avg_Loss=0.0
        avg_middle=0.0
        avg_photo_loss=0.0
        avg_DFF_1=0.0
        avg_DFF_2=0.0
        avg_DFF_3=0.0

    writer.close()
if __name__=="__main__":
    main()
