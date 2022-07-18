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
        #self.num_img=num_img
        self.DFF_net=DFF_net()
    def forward(self,FS,focus_dists):#left_focal_stack):
        dff_out=self.DFF_net(FS,focus_dists)
        return dff_out

class DFF_net(nn.Module):
    def __init__(self):
        super(DFF_net,self).__init__()
        self.FM_measure=FM_module()      #self.RDF_module=self.RDF_module 
        self.FM_conv1 = nn.Sequential(
                        res_stride_conv_3d(8,16),
                        Feature_Extraction(16)
        )
    
        self.FM_conv2 = nn.Sequential(
                        res_stride_conv_3d(16,32),
                        Feature_Extraction(32)
        )

        self.SPP_module = hourglassup(32)
        #self.aggregation = aggregation_module(32)
        self.confidence = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))
        #self.pooling_4 = nn.AvgPool3d((1,4, 4), stride=(1,4,4))
        #self.pooling_2 = nn.AvgPool3d((1,2, 2), stride=(1,2,2))

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
        #self.UNet=nn.DataParallel(self.UNet)
    def forward(self,FS,focus_dists):
            #self.RDF_module=self.RDF_module 
        B,N,C,Height,Width=FS.shape

        FM_result=self.FM_measure(FS)
        half_volume = self.FM_conv1(FM_result)

        quad_volume = self.FM_conv2(half_volume)

        FS_volume=self.SPP_module(quad_volume)
        #input_volume_left=self.aggregation(input_volume_left+med_left)
        #input_volume_right=self.aggregation(input_volume_right+med_right)
        FS_mid_out = self.confidence(FS_volume)#1/8 size
        FS_mid_out = torch.squeeze(FS_mid_out,1)

        FS_mid_out = F.upsample(FS_mid_out,[Height,Width], mode='bilinear')
        mid_out = F.softplus(FS_mid_out)+ 1e-6

        mid_out =  mid_out / mid_out.sum(axis=1,keepdim=True)

        #left_mid_out=focus_dists*self.softmax(left_mid_out*sharpness_scaler)
        mid_out=torch.sum(focus_dists*mid_out,dim=1)
        #right_mid_out=focus_dists*self.softmax(right_mid_out*sharpness_scaler)
        x = FS_volume.contiguous()
        x = self.dres0(x)
        x = self.deconv_1(x)
        out, pre= self.dres2( torch.cat([x,quad_volume],dim=1), None, None) 
        out_in = x + out
        cost1 = self.classif1(out_in)
        
        out2 = self.deconv_2(out_in)

        out, pre = self.dres3(torch.cat([out2,half_volume],dim=1), pre, out) 
        out_in = out2 + out
        cost2 = self.classif2(out_in)
        
        out2 = self.deconv_3(out_in)
        out, _ = self.dres4(torch.cat([out2,FM_result],dim=1), pre, out)
        out = out2 + out
        cost3 = self.classif3(out)

        cost1 = torch.squeeze(cost1,dim=1) 
        cost1 = F.upsample(cost1, [Height,Width], mode='bilinear')
        cost2 = torch.squeeze(cost2,dim=1) 
        cost2 = F.upsample(cost2, [Height,Width], mode='bilinear')
        cost3 = torch.squeeze(cost3,dim=1) 
        pred1 = F.softplus(cost1)+ 1e-6
        pred1 =  pred1 / pred1.sum(axis=1,keepdim=True)
        pred1=torch.sum(focus_dists*pred1,dim=1)

        pred2 = F.softplus(cost2) + 1e-6
        pred2 =  pred2 / pred2.sum(axis=1,keepdim=True)
        pred2=torch.sum(focus_dists*pred2,dim=1)
        
        pred3 = F.softplus(cost3)+ 1e-6

        pred3 = pred3 / pred3.sum(axis=1,keepdim=True)


        pred3=torch.sum(focus_dists*pred3,dim=1)
        return mid_out,  pred1, pred2, pred3



class FM_module(nn.Module):
    def __init__(self):
        super(FM_module, self).__init__()
        self.Focus_extraction = nn.Sequential(
            convbn_3d(3,8,kernel_size=(1,9,9),stride=1,pad=(0,8,8),dilation=(1,2,2)),
            nn.ReLU(inplace=True),
            Feature_Extraction(8)
        )

    def forward(self,x):
        out = self.Focus_extraction(x)
        
        return out

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class hourglassup(nn.Module):
    def __init__(self, in_channels):
        super(hourglassup, self).__init__()

        self.pooling_32 = nn.AvgPool3d((1,8, 8), stride=(1,8,8))

        self.pooling_16 = nn.AvgPool3d((1,4, 4), stride=(1,4,4))

        self.pooling_8 = nn.AvgPool3d((1,2, 2), stride=(1,2,2))

#mish->relu
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

        conv1 = self.conv1(x_8)#strided conv          #1/8
        conv1 = torch.cat((conv1, x_16), dim=1)   #1/8
        conv1 = self.combine1(conv1)   #1/8
        conv2 = self.conv2(conv1)      #1/8

        conv3 = self.conv3(conv2)#strided conv      #1/16
        conv3 = torch.cat((conv3, x_32), dim=1)   #1/16
        conv3 = self.combine2(conv3)   #1/16
        conv4 = self.conv4(conv3)      #1/16


        conv8 = F.relu(self.conv8(conv4) + self.redir2(conv2), inplace=True)
        conv9 = F.relu(self.conv9(conv8) + self.redir1(x_8), inplace=True)
        return conv9

class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()
        self.conv0 = nn.Sequential(convbn_3d(2*inplanes, inplanes, kernel_size=3, stride=1, pad=(1,1,1)),
                                   nn.ReLU(inplace=True))

        #pre

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
                                   nn.BatchNorm3d(2*inplanes)) #+conv2
        #post

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(2*inplanes, inplanes, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x
    def forward(self, x ,presqu, postsqu):
        pre_1  = self.conv0(x)
        out  = self.conv1(pre_1) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)#post:64
        else:
            pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
            out = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8 pre:64
        else:
            out = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(out)  #in:1/8 out:1/4

        return out, pre_1 # out -> postsqu 
class aggregation_module(nn.Module):
    def __init__(self, inplanes):
        super(aggregation_module, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes, kernel_size=3, stride=(1,2,2), pad=(1,1,1)),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(inplanes, inplanes, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(inplanes, inplanes, kernel_size=3, stride=(1,2,2), pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes, inplanes, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes, inplanes, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),bias=False),
                                   nn.BatchNorm3d(inplanes)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes, inplanes, kernel_size=3, padding=1, output_padding=(0,1,1), stride=(1,2,2),bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x
    def forward(self, x):
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16
        out  = self.conv5(out)
        out  = self.conv6(out + pre)  #in:1/8 out:1/4

        return out+x
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
class resnet_block_3d(nn.Module):
    def __init__(self,in_planes):
        super(resnet_block_3d,self).__init__()
        self.conv = nn.Sequential(convbn_3d(in_planes,in_planes,(3,3,3),stride=(1,1,1),pad=(1,1,1),dilation=(1,1,1)),
                   nn.ReLU(inplace=True),
                   convbn_3d(in_planes,in_planes,(3,3,3),stride=(1,1,1),pad=(1,1,1),dilation=(1,1,1)),
        )
        self.Relu = nn.ReLU(inplace=True)

    def forward(self,x):
        return self.Relu(x + self.conv(x))

class res_stride_conv_3d(nn.Module):
    def __init__(self,in_planes,out_planes):
        super(res_stride_conv_3d,self).__init__()
        self.stride_conv = convbn_3d(in_planes,out_planes,(3,3,3),(1,2,2),(1,1,1))
        self.max_pooling = nn.Sequential(nn.MaxPool3d((1,2,2),(1,2,2)),
                                         convbn_3d(in_planes,out_planes,(3,3,3),1,(1,1,1))
        )
        self.RELU=nn.ReLU(inplace=True)
    def forward(self,x):
        return self.RELU(self.stride_conv(x) + self.max_pooling(x))

class Feature_Extraction(nn.Module):
    def __init__(self,in_planes):
        super(Feature_Extraction,self).__init__()
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
        