""" Full assembly of the parts to form the complete network """
import torch.nn as nn
from unet_parts import *
import torch

class CloudUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(CloudUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)

        factor = 2 if bilinear else 1
        self.up3 = Up(256, 128 // factor, bilinear,dp=True)
        self.up4 = Up(128, 64, bilinear,dp=True)
        self.outc = OutConv(64, n_classes)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.dropout(x1)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x = self.up3(x3, x2)

        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__=="__main__":
    model = CloudUNet(n_channels=2,n_classes=2)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: {:,}".format(num_params))
    x = torch.rand(1, 2, 64, 64)
    print(model(x).shape)
    print("Done!")