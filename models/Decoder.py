import torch

import torch.nn as nn


from models.Attention import EdgeEnhance,SpatialAttention,ChannelAttention

class MSFABlock(nn.Module):
    def __init__(self,in_channels,kernel,sample1=None,sample2=None):
        super().__init__()
        self.sample1=sample1
        self.sample2=sample2
        self.extract=nn.Sequential(
            nn.Conv2d(in_channels,in_channels//4,kernel,padding=kernel//2,groups=in_channels//4),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels//4,in_channels//4,1),
            nn.BatchNorm2d(in_channels//4),
        )

    def forward(self,x):
        if self.sample1!=None:
            x=self.sample1(x)
        x=self.extract(x)
        if self.sample2!=None:
            x=self.sample2(x)
        return x

class EAS(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.extract=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,3,padding=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.local=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels)
        )
        self.attn=nn.Sequential(
            EdgeEnhance(in_channels),
            SpatialAttention(in_channels),
            ChannelAttention(in_channels)
        )

    def forward(self,x):
        x=self.extract(x)
        x=self.local(x)
        x=self.attn(x)
        return x

class MSFA(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.c=in_channels
        self.cf1=MSFABlock(in_channels,3)
        self.cf2=MSFABlock(in_channels,5)
        self.cf3=MSFABlock(in_channels,7)
        self.cf4=MSFABlock(in_channels,9)
        

    def forward(self,x):
        x1=self.cf1(x)
        x2=self.cf2(x)
        x3=self.cf3(x)
        x4=self.cf4(x)
        out=torch.cat([x1,x2,x3,x4],dim=1)
        return out
    

class CSAD(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.msfa=MSFA(in_channels)
        self.eas=EAS(in_channels)

    def forward(self,x):
        short_cut=x
        x=self.msfa(x)
        x=self.eas(x)+short_cut
        return x


class CAF(nn.Module):
    def __init__(self,in_channels_e,in_channels_d):
        super().__init__()
        self.pro_Q=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels_d,in_channels_e,1),
            nn.BatchNorm2d(in_channels_e)
        )
        self.pro_K=nn.Sequential(
            nn.Conv2d(in_channels_e,in_channels_e,1),
            nn.BatchNorm2d(in_channels_e)
        )
        self.norm=nn.BatchNorm2d(in_channels_e)
        self.soft=nn.Softmax(-1)
        self.local=nn.Sequential(
            nn.Conv2d(in_channels_e,in_channels_e,1),
            nn.BatchNorm2d(in_channels_e),
            nn.ReLU()
        )
    
    
    
    def forward(self,saliency,details):
        B,C,H,W=saliency.shape
        N=H*W
        saliency_K=self.pro_K(saliency)
        details_Q=self.pro_Q(details)
        saliency_K=saliency_K.view(B,C,H*W)#B,C,N
        details_Q=details_Q.view(B,C,H*W)#B,C,N
        saliency_V=saliency_K
        details_V=details_Q
        attn=details_Q@saliency_K.transpose(1,2)*(N**-0.5)#B,C,C
        attn_score=self.soft(attn.view(B,C*C)).view(B,C,C)#B,C,C
        attn_s=attn_score@saliency_V+saliency_V#B,C,N
        attn_d=attn_score@details_V+details_V#B,C,N
        x=attn_s+attn_d#B,C,N
        x=self.local(x.view(B,C,H,W))#B,C,H,W
        return x


class DecoderBlock(nn.Module):
    def __init__(self,in_channel_e,in_channel_d=None):
        super().__init__()
        if in_channel_d!=None:
            self.csad=CSAD(in_channel_e)
            self.caf=CAF(in_channel_e,in_channel_d)
        else:
            self.csad=CSAD(in_channel_e)
            self.caf=None
        
    def forward(self,x_e,x_d=None):
        if x_d==None:
            x=self.csad(x_e)
        else:
            x_caf=self.caf(x_e,x_d)
            x=self.csad(x_caf)
        return x
    


class Decoder(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512]):
        super().__init__()
        self.db5=DecoderBlock(in_channels[4])
        self.db4=DecoderBlock(in_channels[3],in_channels[4])
        self.db3=DecoderBlock(in_channels[2],in_channels[3])
        self.db2=DecoderBlock(in_channels[1],in_channels[2])
        self.db1=DecoderBlock(in_channels[0],in_channels[1])
        
    def forward(self,x):
        x1,x2,x3,x4,x5=x
        x_list=[]
        x5=self.db5(x5)
        x_list.append(x5)
        x4=self.db4(x4,x5)
        x_list.append(x4)
        x3=self.db3(x3,x4)
        x_list.append(x3)
        x2=self.db2(x2,x3)
        x_list.append(x2)
        x1=self.db1(x1,x2)
        x_list.append(x1)
        return x_list



class PH_Block(nn.Module):
    def __init__(self,in_channels,scale_factor=1):
        super().__init__()
        if scale_factor>1:
            self.upsample=nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        else:
            self.upsample=None
        self.pro=nn.Conv2d(in_channels,1,1)

    def forward(self,x:torch.Tensor):
        if self.upsample!=None:
            x=self.upsample(x)
        x=self.pro(x)
        return x

class PH(nn.Module):
    def __init__(self,in_channels=[64,128,256,512,512],scale_factor=[1,2,4,8,16]):
        super().__init__()
        self.final=nn.ModuleList()
        self.ph1=PH_Block(in_channels[0],scale_factor[0])
        self.ph2=PH_Block(in_channels[1],scale_factor[1])
        self.ph3=PH_Block(in_channels[2],scale_factor[2])
        self.ph4=PH_Block(in_channels[3],scale_factor[3])
        self.ph5=PH_Block(in_channels[4],scale_factor[4])

    def forward(self,x):
        x5,x4,x3,x2,x1=x
        x_list=[]
        x1=self.ph1(x1)
        x_list.append(x1.sigmoid())
        x2=self.ph2(x2)
        x_list.append(x2.sigmoid())
        x3=self.ph3(x3)
        x_list.append(x3.sigmoid())
        x4=self.ph4(x4)
        x_list.append(x4.sigmoid())
        x5=self.ph5(x5)
        x_list.append(x5.sigmoid())
        return x_list


