import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.nn.modules.pooling import MaxPool2d
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.DFF_net=DFF_net()
        self.optical_flow_aggregation =FlowNetwork(8)

    def forward(self,FS,focus_dists,FOVs):
        FS = self.optical_flow_aggregation (FS, FOVs)                
        dff_out=self.DFF_net(FS,focus_dists)
        return dff_out
class FlowNetwork(nn.Module): #Alignment Network
    def __init__(self, inplanes):
        super(FlowNetwork, self).__init__()
        self.OF_feature=nn.Sequential(resnet_block_2d_OF(3,inplanes,1),
                                      resnet_block_2d_OF(inplanes,inplanes,1)
                                      )
        
        self.OF_feature1 = nn.Sequential(resnet_block_2d_OF(inplanes,2*inplanes,2),
                                      resnet_block_2d_OF(2*inplanes,2*inplanes,1)
                                      )
        
        self.OF_feature2 = nn.Sequential(resnet_block_2d_OF(2*inplanes,4*inplanes,2),
                                      resnet_block_2d_OF(4*inplanes,4*inplanes,1)
                                      )

        self.conv1 = nn.Sequential(convbn_3d(8*inplanes+2, 8*inplanes, (1,3,3), stride=1,pad=(0,1,1)),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(8*inplanes, 8*inplanes, (1,3,3), stride=1,pad=(0,1,1)),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(8*inplanes, 8*inplanes, (1,3,3), stride=1,pad=(0,1,1)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(8*inplanes,3, (1,3,3), stride=1,padding=(0,1,1)),
                                    nn.AdaptiveAvgPool3d((10,1,1))
                                    )
        
        self.conv2 = nn.Sequential(convbn_3d(4*inplanes+2, 4*inplanes, (1,3,3), stride=1,pad=(0,1,1)),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(4*inplanes, 4*inplanes, (1,3,3), stride=1,pad=(0,1,1)),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(4*inplanes, 4*inplanes, (1,3,3), stride=1,pad=(0,1,1)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(4*inplanes,3,(1,3,3), stride=1,padding=(0,1,1)),
                                    nn.AdaptiveAvgPool3d((10,1,1))
                                   )

        self.conv3 = nn.Sequential(convbn_3d(2*inplanes+2, 2*inplanes, (1,3,3), stride=1,pad=(0,1,1)),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(2*inplanes, 2*inplanes, (1,3,3), stride=1,pad=(0,1,1)),
                                    nn.ReLU(inplace=True),
                                    convbn_3d(2*inplanes, 2*inplanes, (1,3,3), stride=1,pad=(0,1,1)),
                                    nn.ReLU(inplace=True),
                                    nn.Conv3d(2*inplanes,3, kernel_size=(1,3,3),padding=(0,1,1)),
                                    nn.AdaptiveAvgPool3d((10,1,1))
                                    )

    def forward(self,FS,FOVs):
        FE1 = self.OF_feature(FS)
        FE2 = self.OF_feature1(FE1)
        FE3 = self.OF_feature2(FE2)
        B,C,N,H,W = FE1.shape#


        FE3,flow = self.FOV_warp(FE3,torch.zeros((1,3,1,1)).to(FS.device),FOVs)
        FE3_copy = FE3.new_zeros([B,8*C+2,N,H//4,W//4])
        for i in range(0,N):
            FE3_copy[:,:4*C,i,:,:] = FE3[:,:,-1,:,:]
            FE3_copy[:,4*C:8*C,i,:,:] = FE3[:,:,i,:,:] 
        FE3_copy[:,8*C:8*C+2,:,:,:] = flow
        FE3_copy = FE3_copy.contiguous()

        alpha = self.conv1(FE3_copy)#1/2
        alpha[:,0,:,:]=0.001*alpha[:,0,:,:]
        FE2,flow = self.FOV_warp(FE2,alpha,FOVs)
        FE2_copy = FE2.new_zeros([B,4*C+2,N,H//2,W//2])
        for i in range(0,N):
            FE2_copy[:,:2*C,i,:,:] = FE2[:,:,-1,:,:]
            FE2_copy[:,2*C:4*C,i,:,:] = FE2[:,:,i,:,:] 
        FE2_copy[:,4*C:4*C+2,:,:,:] = flow
        FE2_copy = FE2_copy.contiguous()
        new_alpha = self.conv2(FE2_copy)
        new_alpha[:,0,:,:]=0.001*new_alpha[:,0,:,:].clone()

        alpha = new_alpha + alpha  
        FE1,flow = self.FOV_warp(FE1,alpha,FOVs)
        FE1_copy = FE1.new_zeros([B,2*C+2,N,H,W])
        for i in range(0,N):
            FE1_copy[:,:C,i,:,:] = FE1[:,:,-1,:,:]
            FE1_copy[:,C:2*C,i,:,:] = FE1[:,:,i,:,:] 
        FE1_copy[:,2*C:2*C+2,:,:,:] = flow
        FE1_copy = FE1_copy.contiguous()
        new_alpha = self.conv3(FE1_copy)
        new_alpha[:,0,:,:]=0.001*new_alpha[:,0,:,:].clone()

        alpha = new_alpha + alpha  
        FS,flow = self.FOV_warp(FS,alpha,FOVs)

        return FS
    
    def FOV_warp(self,x, alpha, FOVs): #Warping Function
        B,C,N,H,W = x.shape
        #In paper,
        #alpha : alpha[:,0,:,:]
        #beta : alpha[:,1,:,:]
        #gamma : alpha[[:,2,:,:]
        FOVs = alpha[:,0,:,:] + FOVs
        xx = torch.unsqueeze(torch.unsqueeze(torch.linspace(-1, 1,steps=(W)),dim=0),dim=0).repeat(N,H,1).view(1,1,N,H,W).repeat(B,1,1,1,1)
        yy = torch.unsqueeze(torch.unsqueeze(torch.linspace(-1, 1,steps=(H)),dim=0),dim=2).repeat(N,1,W).view(1,1,N,H,W).repeat(B,1,1,1,1)
        grid2 = torch.cat((xx,yy),1).float().to(x.device)
        
        grid2[:,0,:,:,:] = (W//2)*(FOVs[:,0,:,:,:]-1) * grid2[:,0,:,:,:].clone()  + alpha[:,1,:,:]
        grid2[:,1,:,:,:] = (H//2)*(FOVs[:,0,:,:,:]-1) * grid2[:,1,:,:,:].clone()  + alpha[:,2,:,:]
        #flow unit: pixel
        xx = torch.unsqueeze(torch.unsqueeze(torch.arange(0, W),dim=0),dim=0).repeat(N,H,1).view(1,1,N,H,W).repeat(B,1,1,1,1)
        yy = torch.unsqueeze(torch.unsqueeze(torch.arange(0, H),dim=0),dim=2).repeat(N,1,W).view(1,1,N,H,W).repeat(B,1,1,1,1)
        zz = torch.unsqueeze(torch.unsqueeze(torch.arange(0, N),dim=1),dim=2).repeat(1,H,W).view(1,1,N,H,W).repeat(B,1,1,1,1)
        grid = torch.cat((xx,yy,zz),1).float().to(x.device)
        zero_flow = torch.zeros((B,3,N,H,W),dtype=torch.float32)
        zero_flow[:,:2,:,:,:] = grid2
        zero_flow = zero_flow.to(x.device)
        #flow unit: pixel
        grid = torch.autograd.Variable(grid) - zero_flow.clone()
        grid[:,0,:,:,:] = 2.0*grid[:,0,:,:].clone() / max(W-1,1)-1.0
        grid[:,1,:,:,:] = 2.0*grid[:,1,:,:].clone() / max(H-1,1)-1.0
        grid[:,2,:,:,:] = 2.0*grid[:,2,:,:].clone() / max(N-1,1)-1.0
        grid = grid.permute(0,2,3,4,1).type(torch.float32)#permute(0,2,3,1)          
        output = nn.functional.grid_sample(x, grid,align_corners=True)
        return output, grid2
class resnet_block_2d_OF(nn.Module):
    def __init__(self,in_planes,out_planes,stride):
        super(resnet_block_2d_OF,self).__init__()
        self.conv = nn.Sequential(convbn_3d(in_planes,out_planes,(1,3,3),stride=(1,stride,stride),pad=(0,1,1),dilation=(1,1,1)),
                   nn.ReLU(inplace=True),
                   convbn_3d(out_planes,out_planes,(1,3,3),stride=(1,1,1),pad=(0,1,1),dilation=(1,1,1))
        )
        self.feature = nn.Conv3d(in_planes,out_planes,1,(1,stride,stride),0,bias=False)
        self.relu= nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.feature(x) + self.conv(x))

class DFF_net(nn.Module):
    def __init__(self):
        super(DFF_net,self).__init__()
        #Feature Extraction
        self.FM_measure=FM_module()      
        self.FM_conv1 = nn.Sequential(
                        EFD(8,16),
                        SRD(16)
        )
    
        self.FM_conv2 = nn.Sequential(
                        EFD(16,32),
                        SRD(32)
        )
        #Multi scale Feature Extraction
        self.SPP_module = hourglassup(32)
        
        #Refinment
        self.confidence = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.dres0 = nn.Sequential(convbn_3d(32, 64, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(64, 64, 3, 1, 1),
                                     nn.ReLU(inplace=True))
        self.deconv_1 = nn.Sequential(nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),bias=False),
                                   nn.BatchNorm3d(32)) 
        self.dres2 =hourglass(32)
        self.deconv_2 = nn.Sequential(nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),bias=False),
                                   nn.BatchNorm3d(16)) 
        self.dres3 =hourglass(16)
        self.deconv_3 = nn.Sequential(nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),bias=False),
                                   nn.BatchNorm3d(8)) 
        self.dres4 =hourglass(8)

        self.classif1 = nn.Sequential(nn.Conv3d(32, 1, kernel_size=1, padding=0, stride=1,bias=False))

        self.classif2 = nn.Sequential(nn.Conv3d(16, 1, kernel_size=1, padding=0, stride=1,bias=False))

        self.classif3 = nn.Sequential(nn.Conv3d(8, 1, kernel_size=1, padding=0, stride=1,bias=False))



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self,FS,focus_dists):
        B,N,C,Height,Width=FS.shape
        #Feature Extraction
        FM_result=self.FM_measure(FS)#V1
        half_volume = self.FM_conv1(FM_result)#V2

        quad_volume = self.FM_conv2(half_volume)#V3

        FS_volume=self.SPP_module(quad_volume) #Multi-scale feature aggregation (1st hourglass)
        FS_mid_out = self.confidence(FS_volume)
        FS_mid_out = torch.squeeze(FS_mid_out,1)

        FS_mid_out = F.upsample(FS_mid_out,[Height,Width], mode='bilinear')
        #Softplus Normalization
        mid_out = F.softplus(FS_mid_out)+ 1e-6
        mid_out =  mid_out / mid_out.sum(axis=1,keepdim=True)
        mid_out=torch.sum(focus_dists*mid_out,dim=1) #D1
        
        x = FS_volume.contiguous()
        x = self.dres0(x)
        x = self.deconv_1(x)
        out, pre= self.dres2( torch.cat([x,quad_volume],dim=1), None, None) #2nd hourglass
        out_in = x + out
        cost1 = self.classif1(out_in)
        
        out2 = self.deconv_2(out_in)

        out, pre = self.dres3(torch.cat([out2,half_volume],dim=1), pre, out) #3rd hourglass
        out_in = out2 + out
        cost2 = self.classif2(out_in)
        
        out2 = self.deconv_3(out_in)
        out, _ = self.dres4(torch.cat([out2,FM_result],dim=1), pre, out)#4th hourglass
        out = out2 + out
        cost3 = self.classif3(out)

        cost1 = torch.squeeze(cost1,dim=1) 
        cost1 = F.upsample(cost1, [Height,Width], mode='bilinear')
        cost2 = torch.squeeze(cost2,dim=1) 
        cost2 = F.upsample(cost2, [Height,Width], mode='bilinear')
        cost3 = torch.squeeze(cost3,dim=1) 
        #Softplus Normalization
        pred1 = F.softplus(cost1)+ 1e-6
        pred1 =  pred1 / pred1.sum(axis=1,keepdim=True)
        pred1=torch.sum(focus_dists*pred1,dim=1) # D2
        #Softplus Normalization
        pred2 = F.softplus(cost2) + 1e-6
        pred2 =  pred2 / pred2.sum(axis=1,keepdim=True)
        pred2=torch.sum(focus_dists*pred2,dim=1)#D3
        #Softplus Normalization
        pred3 = F.softplus(cost3)+ 1e-6
        pred3 = pred3 / pred3.sum(axis=1,keepdim=True)
        pred3=torch.sum(focus_dists*pred3,dim=1)#D4
        return mid_out,  pred1, pred2, pred3, FS



class FM_module(nn.Module):
    def __init__(self):
        super(FM_module, self).__init__()
        self.Focus_extraction = nn.Sequential(
            convbn_3d(3,8,kernel_size=(1,9,9),stride=1,pad=(0,8,8),dilation=(1,2,2)),#Dilated convolution (the intial of feature extraction)
            nn.ReLU(inplace=True),
            SRD(8)
        )

    def forward(self,x):
        out = self.Focus_extraction(x)
        
        return out

class hourglassup(nn.Module): #Multi-scale features aggregation
    def __init__(self, in_channels):
        super(hourglassup, self).__init__()

        self.pooling_32 = nn.AvgPool3d((1,8, 8), stride=(1,8,8))

        self.pooling_16 = nn.AvgPool3d((1,4, 4), stride=(1,4,4))

        self.pooling_8 = nn.AvgPool3d((1,2, 2), stride=(1,2,2))

        self.dres8_0 = nn.Sequential(convbn_3d(in_channels, in_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(in_channels, in_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres8_1 = nn.Sequential(convbn_3d(in_channels, in_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(in_channels, in_channels, 3, 1, 1))

        self.dres16_0 = nn.Sequential(convbn_3d(in_channels, in_channels*2, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(in_channels*2, in_channels*2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres16_1 = nn.Sequential(convbn_3d(in_channels*2, in_channels*2, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(in_channels*2, in_channels*2, 3, 1, 1))

        self.dres32_0 = nn.Sequential(convbn_3d(in_channels, in_channels*2, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(in_channels*2, in_channels*2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres32_1 = nn.Sequential(convbn_3d(in_channels*2, in_channels*2, 3, 1, 1),
                                   nn.ReLU(inplace=True),#
                                   convbn_3d(in_channels*2, in_channels*2, 3, 1, 1))
        
        self.conv1 = nn.Conv3d(in_channels, in_channels * 2, kernel_size=3, stride=(1,2,2),
                                   padding=1, bias=False)

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Conv3d(in_channels * 2, in_channels * 4, kernel_size=3, stride=(1,2,2),
                               padding=1, bias=False)

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=(0,1,1), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(in_channels))

        self.combine1 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.combine2 = nn.Sequential(convbn_3d(in_channels * 6, in_channels * 4, 3, 1, 1),
                                      nn.ReLU(inplace=True))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.redir3 = convbn_3d(in_channels * 4, in_channels * 4, kernel_size=1, stride=1, pad=0)


    def forward(self, x):
        x_32=self.pooling_32(x)
        x_16=self.pooling_16(x)
        x_8=self.pooling_8(x)
        x_8_res=self.dres8_0(x_8)
        x_8 = self.dres8_1(x_8_res) +x_8_res

        x_16_res=self.dres16_0(x_16)
        x_16 = self.dres16_1(x_16_res) +x_16_res

        x_32_res=self.dres32_0(x_32)
        x_32 = self.dres32_1(x_32_res) +x_32_res

        conv1 = self.conv1(x_8)
        conv1 = torch.cat((conv1, x_16), dim=1)   
        conv1 = self.combine1(conv1)  
        conv2 = self.conv2(conv1)      

        conv3 = self.conv3(conv2)
        conv3 = torch.cat((conv3, x_32), dim=1)   
        conv3 = self.combine2(conv3)   
        conv4 = self.conv4(conv3)      


        conv8 = F.relu(self.conv8(conv4) + self.redir2(conv2), inplace=True)
        conv9 = F.relu(self.conv9(conv8) + self.redir1(x_8), inplace=True)
        return conv9

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()
        self.conv0 = nn.Sequential(convbn_3d(2*inplanes, inplanes, kernel_size=3, stride=1, pad=(1,1,1)),
                                   nn.ReLU(inplace=True))


        self.conv1 = nn.Sequential(convbn_3d(inplanes, 2*inplanes, kernel_size=3, stride=(1,2,2), pad=(1,1,1)),
                                   nn.ReLU(inplace=True))
        self.pre_conv =nn.Sequential(convbn_3d(2*inplanes, 2*inplanes, kernel_size=1, stride=1,pad=0),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(2*inplanes, 2*inplanes, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(2*inplanes, 2*inplanes, kernel_size=3, stride=(1,2,2), pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(2*inplanes, 2*inplanes, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(2*inplanes, 2*inplanes, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),bias=False),
                                   nn.BatchNorm3d(2*inplanes)) 

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(2*inplanes, inplanes, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),bias=False),
                                   nn.BatchNorm3d(inplanes))
    def forward(self, x ,presqu, postsqu):
        pre_1  = self.conv0(x)
        out  = self.conv1(pre_1)
        pre  = self.conv2(out) 
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) 
        out  = self.conv4(out) 

        if presqu is not None:
            out = F.relu(self.conv5(out)+presqu, inplace=True)
        else:
            out = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(out)

        return out, pre_1 
    
def convbn_3d(in_planes, out_planes, kernel_size, stride, pad,dilation=1):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, dilation=dilation, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))
class resnet_block_2d(nn.Module):
    def __init__(self,in_planes):
        super(resnet_block_2d,self).__init__()
        self.conv = nn.Sequential(convbn_3d(in_planes,in_planes,(1,3,3),stride=(1,1,1),pad=(0,1,1),dilation=(1,1,1)),
                   nn.ReLU(inplace=True),
                   convbn_3d(in_planes,in_planes,(1,3,3),stride=(1,1,1),pad=(0,1,1),dilation=(1,1,1)),
        )
        self.Relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.Relu((x) + self.conv(x))

class EFD(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(EFD,self).__init__()
        self.stride_conv = convbn_3d(in_planes,out_planes,(3,3,3),(1,2,2),(1,1,1))
        self.max_pooling = nn.Sequential(nn.MaxPool3d((1,2,2),(1,2,2)),
                                         convbn_3d(in_planes,out_planes,(3,3,3),1,(1,1,1))
        )
        self.RELU=nn.ReLU(inplace=True)
    def forward(self,x):
        return self.RELU(self.stride_conv(x) + self.max_pooling(x))

class SRD(nn.Module):
    def __init__(self,in_planes):
        super(SRD,self).__init__()
        self.Focus_Measure =resnet_block_2d(in_planes)
        self.N_ch_attention = nn.Sequential(
            nn.Conv3d(in_planes,in_planes,(3,1,1),stride=1,padding=(1,0,0),bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_planes,in_planes,1,stride=1,padding=0,bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        Feature = self.Focus_Measure(x)
        return Feature + self.N_ch_attention(Feature)
        