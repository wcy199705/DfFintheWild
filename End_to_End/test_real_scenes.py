import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2

from torch.utils.data import DataLoader
from Test_dataloader import Real_Scenes
from tqdm import tqdm

from End_to_End import Network
from matplotlib import cm
from imageio import imwrite
def main():
    model=Network()
    test_dataset=Real_Scenes()
    int_num_imgs=10
    model=Network()
    model=model.cpu()
    model = nn.DataParallel(model)
    model.module.load_state_dict(torch.load('check_point.pth'))
    model=model.cuda()
    dataloader=DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=1)
    model.eval()
    with torch.no_grad():

        for idx,samples in enumerate(tqdm(dataloader,desc="Valid")):
            valid_input , test_focus_dists, test_FOV = samples
            test_focus_dists=test_focus_dists.cuda()
            test_FOV=test_FOV.cuda()
            valid_input=valid_input.cuda()   

            _,_, _, test_depth, test_warp_FS = model(valid_input,test_focus_dists,test_FOV)
            test_depth=test_depth.data.cpu().numpy()#[0,29]

            test_depth = test_depth[0,:-16,:]

            test_focus_dists = test_focus_dists.data.cpu().numpy()
            test_depth = (test_depth - np.min(test_depth))/(np.max(test_depth)-np.min(test_depth))
            test_warp_FS = test_warp_FS.data.cpu().numpy()
            test_warp_FS = np.squeeze(127.5* (test_warp_FS+1.0)).astype(np.uint8)
            test_warp_FS = np.transpose(test_warp_FS,(2,3,0,1))[:,:,:,:]
            for i in range(0,int_num_imgs):
                cv2.imwrite('test/warped_result/' + str(idx) +"/" + str(i) +".png",test_warp_FS[:-16,:,:,i])

            cmap = cm.get_cmap('jet')
            color_img = cmap(
                (test_depth))[..., :3]
            imwrite('test/depth/'+str(idx)+'.jpg', color_img[:, :], quality=100)

        
if __name__=="__main__":
    main()
