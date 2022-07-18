import numpy as np
import torch.nn as nn
import torch
import time
import gc
from Depth_Estimation_Network import Network
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from train_Dataloader import HCI_dataset
from tqdm import tqdm
from metrics import *
import argparse

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
    writer = SummaryWriter(log_dir=root+'logs')
    mid_weight=0.3

    Weight1=0.5
    Weight2=0.7
    Weight3=1.0
    batch_size=args.batch_size

    max_epoch=args.max_epoch
    test_epoch=10 
    save_epoch=10
    load_epoch=args.load_epoch
    print_epoch=10
    model=Network()
    model=model.cpu()
    max_Depth = 2.5
    min_Depth = -2.5
    dataroot='Datasets/HCI/HCI_FS_trainval.h5'

    train_dataset=HCI_dataset(dataroot,"stack_train","disp_train")
    valid_dataset=HCI_dataset(dataroot,"stack_val","disp_val")
    num_train=20
    num_val=4
    avg_Loss=0.0
    avg_middle=0.0
    avg_DFF_1=0.0
    avg_DFF_2=0.0
    avg_DFF_3=0.0
    start=time.time()
    model = nn.DataParallel(model)
    if(load_epoch>1):
        path=root+'models/'+str(load_epoch)+'.pth'
        model.load_state_dict(torch.load(path))
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.99))
    model=model.cuda()
    #scaler=GradScaler()
    dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=args.cpus,pin_memory=True)
    valid_dataloader=DataLoader(valid_dataset,1,shuffle=False,num_workers=4,pin_memory=True)
    #amp
    for epoch in range(load_epoch,max_epoch+1):#chang validation part
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

        if(epoch%save_epoch==0 and epoch!=load_epoch):
            path=root+'models/' + str(epoch)+'.pth'
            torch.save(model.state_dict(),path)
        #validation
        if(epoch%test_epoch==0 and epoch !=load_epoch):
            model.eval()
            with torch.no_grad():
                Avg_mse=0.0
                Avg_mae=0.0
                Avg_rmse=0.0
                Avg_Bulmp=0.0

                val_time=0.0
                for idx, samples in enumerate(tqdm(valid_dataloader,desc="valid")):
                    valid_input, test_gt_depth , test_focus_dists, test_mask = samples
                    test_gt_depth=test_gt_depth.numpy()
                    test_mask = np.squeeze(test_mask.data.cpu().numpy())
                    test_gt_depth = np.squeeze(test_gt_depth)
                    test_focus_dists=test_focus_dists.cuda()   
                    start= time.time()            
                    _,_, _, test_pred3 = model(valid_input,test_focus_dists)
                    val_time = val_time+ (time.time() -start)
                    test_pred3=test_pred3.data.cpu().numpy()#[0,29]
                    test_pred3=np.squeeze(test_pred3)



                    Avg_mse = Avg_mse + mask_mse(test_pred3,test_gt_depth,test_mask)
                    Avg_mae = Avg_mae + mask_mae(test_pred3,test_gt_depth,test_mask)
                    Avg_Bulmp = Avg_Bulmp +get_bumpiness(test_gt_depth,test_pred3,test_mask)
                    Avg_rmse = Avg_rmse + mask_rmse(test_pred3,test_gt_depth,test_mask)
                print("Avg_mse(" +str(epoch)+") : " ,Avg_mse/num_val)
                print("Avg_mae(" +str(epoch)+") : " ,Avg_mae/num_val)
                print("Avg_Bulmp(" +str(epoch)+") : " ,Avg_Bulmp/num_val)

                print("Avg_rmse(" +str(epoch)+") : " ,Avg_rmse/num_val)

                
                print("AVG_time:",val_time/num_val)
                writer.add_scalar("Loss/validation/DFF/Avg_mse",Avg_mse/num_val,epoch)
                writer.add_scalar("Loss/validation/DFF/Avg_rmse",Avg_rmse/num_val,epoch)
                writer.add_scalar("Loss/validation/DFF/Avg_Bulmp",Avg_Bulmp/num_val,epoch)
                writer.add_scalar("Loss/validation/DFF/Avg_mae",Avg_mae/num_val,epoch)
        #Training session
        model.train()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print (name, param.data.shape)
    # exit()
        for idx, samples in enumerate(tqdm(dataloader,desc="Train")): #check variable ranges, images
            train_input, train_gt_depth , train_focus_dists, train_mask = samples

            train_input=train_input.cuda(non_blocking=True)
            train_gt_depth=train_gt_depth.cuda(non_blocking=True)
            train_focus_dists=train_focus_dists.cuda(non_blocking=True)
            train_mask=train_mask.cuda(non_blocking=True)

            mid_out,pred1, pred2, pred3=model(train_input,train_focus_dists)
            pred1 = (pred1 - min_Depth)/(max_Depth - min_Depth)
            pred2 = (pred2 - min_Depth)/(max_Depth - min_Depth)
            pred3 = (pred3 - min_Depth)/(max_Depth - min_Depth)
            train_gt_depth = (train_gt_depth - min_Depth)/(max_Depth - min_Depth)
            
            
            optimizer.zero_grad()

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
            
            #del Loss1, Loss2, Loss3, Total_Loss, mid_loss#,gt_gradient,gt_sobel
        if(epoch%print_epoch==0 and epoch !=load_epoch):
            print("Epoch:",epoch,"AVG_DFF_TotalLoss:",avg_Loss/(num_train*print_epoch))
            writer.add_scalar("Loss/train/Total loss",avg_Loss/(num_train*print_epoch),epoch)
            writer.add_scalar("Loss/train/Mid loss",avg_middle/(num_train*print_epoch),epoch)
            writer.add_scalar("Loss/train/First/L1 loss",avg_DFF_1/(num_train*print_epoch),epoch)
            writer.add_scalar("Loss/train/Second/L1 loss",avg_DFF_2/(num_train*print_epoch),epoch)
            writer.add_scalar("Loss/train/Third/L1 loss",avg_DFF_3/(num_train*print_epoch),epoch)
            start=time.time()
            avg_Loss=0.0
            avg_middle=0.0

            avg_DFF_1=0.0
            avg_DFF_2=0.0
            avg_DFF_3=0.0



    writer.close()
if __name__=="__main__":
    main()
