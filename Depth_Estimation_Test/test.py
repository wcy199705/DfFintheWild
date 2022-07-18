import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
from matplotlib import cm



from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics import *
from imageio import imwrite
import argparse
from Depth_Estimation_Network import Network 
from test_Dataloader import FS6_dataset,HCI_dataset,DDFF12dataset_benchmark,Smartphone,Middlebury


import imageio.core.util

def silence_imageio_warning(*args, **kwargs):
    pass




def main():
    imageio.core.util._precision_warn = silence_imageio_warning

    model=Network()#RGB: 3channel, img:30
    model=model.cpu()
    model = nn.DataParallel(model)
    start=time.time()
    parser = argparse.ArgumentParser(description='Test code: Learning Depth from focus in the wild')
    parser.add_argument('--dataset',type=str,help='Test dataset')
    args = parser.parse_args()
    if args.dataset == 'DefocusNet':        
        max_depth = 1.5
        min_depth = 0.1
        num_test=100
        Dataset=FS6_dataset()
        root = "Results_test/DefocusNet/"
    elif args.dataset == '4D_Light_Field':        
        max_depth = 2.5
        min_depth = -2.5
        num_test=4
        Dataset=HCI_dataset()
        root = "Results_test/4D_Light_Field/"
    elif args.dataset == 'DDFF':        
        focal_length = 521.4052
        K2 = 1982.0250823695178
        flens = 7317.020641763665
        baseline = K2 / flens * 1e-3
        Height=383
        Width=552
        max_depth = baseline * focal_length / 0.5
        min_depth = baseline * focal_length / 7
        num_test=120
        Dataset=DDFF12dataset_benchmark()
        root = "Results_test/DDFF/"
    elif args.dataset == 'Smartphone': 
        Height = 336
        Width = 252
        Dataset=Smartphone()
        root = "Results_test/Smartphone/"
        num_test=47
    elif args.dataset == 'FlyingThings3D': 
        Dataset=Middlebury()
        Dataset2=FS6_dataset()
        root = "Results_test/FlyingThings3D/"
        save_root1 =root + "Middlebury/"
        save_root2 =root + "DefocusNet/"
        num_test=15
        num_test2=100
        
        
    path=root+'check_point.pth'
    model.module.load_state_dict(torch.load(path))

    model=model.cuda()
    #scaler=GradScaler()
    dataloader=DataLoader(Dataset,1,shuffle=False,num_workers=4,pin_memory=True)
    #amp

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
        for idx, samples in enumerate(tqdm(dataloader,desc="Test")):
            if args.dataset =="DDFF":
                valid_input , test_focus_dists = samples
            elif args.dataset == "Smartphone":
                valid_input, test_gt_depth , test_focus_dists, test_mask, test_conf = samples
                test_gt_depth=test_gt_depth.numpy()
                test_mask = np.squeeze(test_mask.data.cpu().numpy())
                test_conf = np.squeeze(test_conf.data.cpu().numpy())
                test_gt_depth = np.squeeze(test_gt_depth)
                max_depth = np.max(test_gt_depth[test_conf==1.0])
                min_depth = np.min(test_gt_depth[test_conf==1.0])
            else:
                valid_input, test_gt_depth , test_focus_dists, test_mask = samples
            #test_gt_sobel=test_gt_sobel[0]
                test_gt_depth=test_gt_depth.numpy()
                test_mask = np.squeeze(test_mask.data.cpu().numpy())
                test_gt_depth = np.squeeze(test_gt_depth)
            test_focus_dists=test_focus_dists.cuda()   
            valid_input = valid_input.cuda()
            start= time.time()            
            _,_, _, test_pred3 = model(valid_input,test_focus_dists)
            val_time = val_time+ (time.time() -start)

            test_pred3=test_pred3.data.cpu().numpy()#[0,29]
            #cv2.imwrite(root+'Depth/Test/'+str(idx)+'.png',test_pred3_save)
            test_pred3=np.squeeze(test_pred3)
            if args.dataset =="DDFF" or args.dataset=="Smartphone":
                test_pred3 = test_pred3[:Height,:Width]
            if args.dataset =="FlyingThings3D":
                min_depth = 10
                max_depth = 60
                H,W = test_gt_depth.shape
                test_pred3=test_pred3[:H,:W]


            cmap = cm.get_cmap('jet')
            color_img = cmap(
                ((test_pred3 - min_depth) / (max_depth - min_depth)))[..., :3]
            if args.dataset == "FlyingThings3D":
                imwrite(save_root1+'Depth/'+str(idx)+'.jpg', color_img[:, :], quality=100)

            else:
                imwrite(root+'Depth/'+str(idx)+'.jpg', color_img[:, :], quality=100)


            #cv2.imwrite(root+'Depth/GT/'+str(idx)+'.png',gt_depth_255)

            if args.dataset =="DDFF":
                pass
            #io.savemat(root+'ERRORMAP/'+str(idx)+'.mat',{'error':np.squeeze(Error_map_focus)})
            elif args.dataset == "Smartphone":
                Avg_mse = Avg_mse + mask_mse_w_conf(test_pred3,test_gt_depth,test_conf,test_mask)
                Avg_mae = Avg_mae + mask_mae_w_conf(test_pred3,test_gt_depth,test_conf,test_mask)
            else:
                Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_pred3,test_gt_depth,test_mask)
                Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_pred3,test_gt_depth,test_mask)
                Avg_mse = Avg_mse + mask_mse(test_pred3,test_gt_depth,test_mask)
                Avg_mae = Avg_mae + mask_mae(test_pred3,test_gt_depth,test_mask)

                Avg_rmse = Avg_rmse + mask_rmse(test_pred3,test_gt_depth,test_mask)
                Avg_rmse_log = Avg_rmse_log + mask_rmse_log(test_pred3,test_gt_depth,test_mask)
                Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_pred3,test_gt_depth,1,test_mask)
                Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_pred3,test_gt_depth,2,test_mask)
                Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_pred3,test_gt_depth,3,test_mask)

        if args.dataset =="DDFF":
            pass
        elif args.dataset == "Smartphone":
            print("Avg_mse: " ,Avg_mse/num_test)
            print("Avg_mae: " ,Avg_mae/num_test)
        else:
            print("Avg_abs_rel : " ,Avg_abs_rel/num_test)
            print("Avg_sq_rel : " ,Avg_sq_rel/num_test)
            print("Avg_mse : " ,Avg_mse/num_test)
            print("Avg_mae : " ,Avg_mae/num_test)

            print("Avg_rmse : " ,Avg_rmse/num_test)
            print("Avg_rmse_log : " ,Avg_rmse_log/num_test)
            print("Avg_accuracy_1 : " ,Avg_accuracy_1/num_test)
            print("Avg_accuracy_2 : " ,Avg_accuracy_2/num_test)
            print("Avg_accuracy_3 : " ,Avg_accuracy_3/num_test)

            
        print("AVG_time:",val_time/num_test)
        if args.dataset =="FlyingThings3D":
            dataloader=DataLoader(Dataset2,1,shuffle=False,num_workers=4,pin_memory=True)
            Avg_abs_rel=0.0
            Avg_sq_rel=0.0
            Avg_mse=0.0
            Avg_mae=0.0
            Avg_rmse=0.0
            Avg_rmse_log=0.0
            Avg_accuracy_1=0.0
            Avg_accuracy_2=0.0
            Avg_accuracy_3=0.0

            for idx, samples in enumerate(tqdm(dataloader,desc="Test2")):
                valid_input, test_gt_depth , test_focus_dists, test_mask = samples
            #test_gt_sobel=test_gt_sobel[0]
                test_gt_depth=test_gt_depth.numpy()
                test_mask = np.squeeze(test_mask.data.cpu().numpy())
                test_gt_depth = np.squeeze(test_gt_depth)
                test_focus_dists=test_focus_dists.cuda()   
                valid_input = valid_input.cuda()
                start= time.time()            
                _,_, _, test_pred3 = model(valid_input,test_focus_dists)
                val_time = val_time+ (time.time() -start)

                test_pred3=test_pred3.data.cpu().numpy()#[0,29]
                #cv2.imwrite(root+'Depth/Test/'+str(idx)+'.png',test_pred3_save)
                test_pred3=np.squeeze(test_pred3)
                max_depth = 1.5
                min_depth = 0.1


                cmap = cm.get_cmap('jet')
                color_img = cmap(
                    ((test_pred3 - min_depth) / (max_depth - min_depth)))[..., :3]
                imwrite(save_root2+'Depth/'+str(idx)+'.jpg', color_img[:, :], quality=100)


                #cv2.imwrite(root+'Depth/GT/'+str(idx)+'.png',gt_depth_255)

                Avg_abs_rel = Avg_abs_rel + mask_abs_rel(test_pred3,test_gt_depth,test_mask)
                Avg_sq_rel = Avg_sq_rel + mask_sq_rel(test_pred3,test_gt_depth,test_mask)
                Avg_mse = Avg_mse + mask_mse(test_pred3,test_gt_depth,test_mask)
                Avg_mae = Avg_mae + mask_mae(test_pred3,test_gt_depth,test_mask)

                Avg_rmse = Avg_rmse + mask_rmse(test_pred3,test_gt_depth,test_mask)
                Avg_rmse_log = Avg_rmse_log + mask_rmse_log(test_pred3,test_gt_depth,test_mask)
                Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(test_pred3,test_gt_depth,1,test_mask)
                Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(test_pred3,test_gt_depth,2,test_mask)
                Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(test_pred3,test_gt_depth,3,test_mask)

            print("Avg_abs_rel : " ,Avg_abs_rel/num_test2)
            print("Avg_sq_rel : " ,Avg_sq_rel/num_test2)
            print("Avg_mse : " ,Avg_mse/num_test2)
            print("Avg_mae : " ,Avg_mae/num_test2)

            print("Avg_rmse : " ,Avg_rmse/num_test2)
            print("Avg_rmse_log : " ,Avg_rmse_log/num_test2)
            print("Avg_accuracy_1 : " ,Avg_accuracy_1/num_test2)
            print("Avg_accuracy_2 : " ,Avg_accuracy_2/num_test2)
            print("Avg_accuracy_3 : " ,Avg_accuracy_3/num_test2)

        
        #Training session
if __name__=="__main__":
    main()
