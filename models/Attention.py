import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super().__init__()

        self.channel=nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        weight=self.channel(x)#B,C,1,1
        x=weight*x+x
        return x



class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.weight=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,7,padding=3,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels,in_channels,3,padding=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
    

    def forward(self, x):
        weight=self.weight(x)#B,1,H,W
        x=weight*x+x
        return x


class EdgeEnhance(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.context=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,7,padding=3,groups=in_channels),
            nn.BatchNorm2d(in_channels),
        )

        self.AP=nn.AvgPool2d(3,1,1)

        self.conv=nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self,x):
        x_ap=self.AP(self.context(x))
        weight=self.conv(x-x_ap)#B,C,H,W
        x_ee=weight*x+x
        return x_ee
