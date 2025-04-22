import torch.nn as nn

from models.Decoder import Decoder,PH
from models.Encoder import Encoder

class Model(nn.Module):
    def __init__(self,input_channels=3, out_channels=[8,16,24,32,40],scale_factor=[1,2,4,8,16],adjust=0.5) -> None:
        super().__init__()
        self.encoder=Encoder(input_channels,out_channels,adjust=adjust)
        self.decoder=Decoder(out_channels)
        self.ph=PH(out_channels,scale_factor)

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        x=self.ph(x)
        return x