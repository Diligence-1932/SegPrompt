import torch
from torch import nn
from mamba_ssm import Mamba

class HPE(nn.Module):
    
    def __init__(self, input_channels=3, out_channels=3,pool=True,adjust=0.5):
        super().__init__()
        if pool:
            self.pool=nn.MaxPool2d(2, stride=2)
        else:
            self.pool=None
        self.ipc=nn.Sequential(
            nn.Conv2d(input_channels,out_channels,7,padding=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.mamba=Mamba(
                d_model=out_channels, # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,    # Local convolution width
                expand=2,    # Block expansion factor
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(out_channels,out_channels,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.prompt1=nn.Conv2d(out_channels,1,1)
        self.norm=nn.LayerNorm(1)
        self.prompt2=nn.Conv2d(out_channels,1,1)
        self.adjust=adjust
    
    def forward(self, x):
        if self.pool!=None:
            x=self.pool(x)
        x=self.ipc(x)
        B,C,H,W=x.shape
        N=H*W
        p1=self.prompt1(x)
        p1=self.norm(p1.view(B,1,N).permute(0,2,1).contiguous()).permute(0,2,1).contiguous().view(B,1,H,W)
        x=self.adjust*self.mamba(x.view(B,C,N).permute(0,2,1).contiguous()).permute(0,2,1).contiguous().view(B,C,H,W)+(1-self.adjust)*x
        p2=self.prompt2(x)
        p2=self.norm(p2.view(B,1,N).permute(0,2,1).contiguous()).permute(0,2,1).contiguous().view(B,1,H,W)
        x=self.conv4(x)+p1+p2
        return x


        

class Encoder(nn.Module):
    
    def __init__(self, input_channels=3, out_channels=[8,16,24,32,40,48],adjust=0.5):
        super().__init__()
        self.block1=HPE(input_channels,out_channels[0],pool=False,adjust=adjust)
        self.block2=HPE(out_channels[0],out_channels[1],adjust=adjust)
        self.block3=HPE(out_channels[1],out_channels[2],adjust=adjust)
        self.block4=HPE(out_channels[2],out_channels[3],adjust=adjust)
        self.block5=HPE(out_channels[3],out_channels[4],adjust=adjust)

    def forward(self, x):
        x_list=[]
        x=self.block1(x)
        x_list.append(x)
        x=self.block2(x)
        x_list.append(x)
        x=self.block3(x)
        x_list.append(x)
        x=self.block4(x)
        x_list.append(x)
        x=self.block5(x)
        x_list.append(x)
        return x_list
        