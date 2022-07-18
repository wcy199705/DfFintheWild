import numpy as np
import torch.nn as nn
import torch
import time
import gc
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from train_Dataloader import FlyingThings3d
from tqdm import tqdm
from metrics import *
import argparse
from Depth_Estimation_Network import Network

MSE_loss = nn.MSELoss()

def masked_MSE_loss(est,gt,mask):
    out = MSE_loss(est[mask],gt[mask])
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
    train_dataset=FlyingThings3d('train')
    valid_dataset=FlyingThings3d('val')
    writer = SummaryWriter(log_dir=root+'logs')
    mid_weight=0.3

    Weight1=0.5
    Weight2=0.7
    Weight3=1.0
#DISP test add..
    int_num_imgs=15
    #num_imgs=50.0
    #int_num_imgs=50
    batch_size=args.batch_size

    max_epoch=2500
    valid_epoch=1 #만들기.
    save_epoch=1
    load_epoch=args.load_epoch#75
    model=Network()#RGB: 3channel, img:30
    model=model.cpu()
    valid_max_depth = 100.0
    valid_min_depth =10.0

    input_size=(540,960)

    num_train=1000
    num_val=100
    avg_Loss=0.0
    avg_middle=0.0
    avg_DFF_1=0.0
    avg_DFF_2=0.0
    avg_DFF_3=0.0
    start=time.time()
    model = nn.DataParallel(model)

    if(load_epoch>1):
        path=root+'models/'+str(load_epoch)+'.pth'
        model.module.load_state_dict(torch.load(path))
        #model_Defocus.load_state_dict(torch.load(path))
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.99))#hyperparmeters from AAAI2020 lr 0.00001 , betas=(0,0.9) #70 epoch 0.001->0.0001
    model=model.cuda()
    #scaler=GradScaler()
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
        #validation
        if(epoch%valid_epoch==0 and epoch !=load_epoch):
            model.eval()
            with torch.no_grad():
                Avg_abs_rel=0.0
                Avg_sq_rel=0.0
                Avg_mse=0.0
                Avg_mae=0.0
                Avg_rmse=0.0
                Avg_rmse_log=0.0
                Avg_accuracy_1=0.0
                Avg_accuracy_2=0.0
                Avg_accuracy_3=0.0

                val_time=0.0
                for idx, samples in enumerate(tqdm(valid_dataloader,desc="valid")):
                    valid_input, test_gt_depth , test_mask, test_focus_dists = samples
                    test_gt_depth=test_gt_depth.numpy()
                    test_mask = np.squeeze(test_mask.data.cpu().numpy())
                    test_gt_depth = np.squeeze(test_gt_depth)
                    test_focus_dists=test_focus_dists.cuda()   
                    start= time.time()            
                    _,_, _, test_pred3 = model(valid_input,test_focus_dists)
                    val_time = val_time+ (time.time() -start)
                    test_pred3=test_pred3.data.cpu().numpy()#[0,29]
                    
                    test_pred3=test_pred3[0,:input_size[0],:]

                    test_pred3=np.squeeze(test_pred3)


                    Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_pred3,test_gt_depth,test_mask)
                    Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_pred3,test_gt_depth,test_mask)
                    Avg_mse = Avg_mse + mask_mse(test_pred3,test_gt_depth,test_mask)
                    Avg_mae = Avg_mae + mask_mae(test_pred3,test_gt_depth,test_mask)

                    Avg_rmse = Avg_rmse + mask_rmse(test_pred3,test_gt_depth,test_mask)
                    Avg_rmse_log = Avg_rmse_log + mask_rmse_log(test_pred3,test_gt_depth,test_mask)
                    Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_pred3,test_gt_depth,1,test_mask)
                    Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_pred3,test_gt_depth,2,test_mask)
                    Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_pred3,test_gt_depth,3,test_mask)
                print("Avg_abs_rel(" +str(epoch)+") : " ,Avg_abs_rel/num_val)
                print("Avg_sq_rel(" +str(epoch)+") : " ,Avg_sq_rel/num_val)
                print("Avg_mse(" +str(epoch)+") : " ,Avg_mse/num_val)
                print("Avg_mae(" +str(epoch)+") : " ,Avg_mae/num_val)

                print("Avg_rmse(" +str(epoch)+") : " ,Avg_rmse/num_val)
                print("Avg_rmse_log(" +str(epoch)+") : " ,Avg_rmse_log/num_val)
                print("Avg_accuracy_1(" +str(epoch)+") : " ,Avg_accuracy_1/num_val)
                print("Avg_accuracy_2(" +str(epoch)+") : " ,Avg_accuracy_2/num_val)
                print("Avg_accuracy_3(" +str(epoch)+") : " ,Avg_accuracy_3/num_val)

                
                print("AVG_time:",val_time/num_val)
                writer.add_scalar("Loss/validation/DFF/Avg_abs_rel",Avg_abs_rel/num_val,epoch)
                writer.add_scalar("Loss/validation/DFF/Avg_sq_rel",Avg_sq_rel/num_val,epoch)
                writer.add_scalar("Loss/validation/DFF/Avg_mse",Avg_mse/num_val,epoch)
                writer.add_scalar("Loss/validation/DFF/Avg_rmse",Avg_rmse/num_val,epoch)
                writer.add_scalar("Loss/validation/DFF/Avg_rmse_log",Avg_rmse_log/num_val,epoch)
                writer.add_scalar("Loss/validation/DFF/Avg_mae",Avg_mae/num_val,epoch)

                writer.add_scalar("Loss/validation/DFF/Avg_accuracy_1",Avg_accuracy_1/num_val,epoch)
                writer.add_scalar("Loss/validation/DFF/Avg_accuracy_2",Avg_accuracy_2/num_val,epoch)
                writer.add_scalar("Loss/validation/DFF/Avg_accuracy_3",Avg_accuracy_3/num_val,epoch)                
        #Training session
        model.train()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data.shape)
    # exit()
        for idx, samples in enumerate(tqdm(dataloader,desc="Train")): #check variable ranges, images
            train_input, train_gt_depth , train_mask, train_focus_dists = samples

            train_input=train_input.cuda(non_blocking=True)
            train_gt_depth=train_gt_depth.cuda(non_blocking=True)
            train_focus_dists=train_focus_dists.cuda(non_blocking=True)
            train_mask=train_mask.cuda(non_blocking=True)

            mid_out,pred1, pred2, pred3=model(train_input,train_focus_dists)
            optimizer.zero_grad()
            
            mid_out = (mid_out - valid_min_depth) / (valid_max_depth - valid_min_depth)
            pred1 = (pred1 - valid_min_depth) / (valid_max_depth - valid_min_depth)
            pred2 = (pred2 - valid_min_depth) / (valid_max_depth - valid_min_depth)
            pred3 = (pred3 - valid_min_depth) / (valid_max_depth - valid_min_depth)
            train_gt_depth = (train_gt_depth - valid_min_depth) / (valid_max_depth - valid_min_depth)

            Loss1 =masked_MSE_loss(pred1,train_gt_depth,train_mask)#,gt_gradient,gt_sobel)
            Loss2 =masked_MSE_loss(pred2,train_gt_depth,train_mask)#,gt_gradient,gt_sobel)
            Loss3 =masked_MSE_loss(pred3,train_gt_depth,train_mask)#,gt_gradient,gt_sobel)

            mid_loss = masked_MSE_loss(mid_out,train_gt_depth,train_mask)
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

        start=time.time()
        avg_Loss=0.0
        avg_middle=0.0

        avg_DFF_1=0.0
        avg_DFF_2=0.0
        avg_DFF_3=0.0

    writer.close()
if __name__=="__main__":
    main()
