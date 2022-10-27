'''
    Author: Zahid Hassan Tushar
    email: ztushar1@umbc.edu

    Write Comment
'''

# Import Libraries
import torch.nn as nn
from torchvision import transforms
from torchinfo import summary

class DNN2w(nn.Module):
    
    def __init__(self):
        super().__init__()

        ### Convolutional layer
        self.conv1      = nn.Conv2d(in_channels=2,out_channels=100, 
        kernel_size=5, stride=1, padding=0)

        self.conv2      = nn.Conv2d(in_channels=100,out_channels=2, 
        kernel_size=1, stride=1, padding=0)

        ### Non-Linear Activation
        self.activation = nn.ReLU()

        ### Fully Connected layer
        self.fc1a = nn.Linear(in_features=6*6*2, out_features=1024)
        self.fc1b = nn.Linear(in_features=6*6*2, out_features=1024)

        self.fc2a = nn.Linear(in_features=1024, out_features=6*6)
        self.fc2b = nn.Linear(in_features=1024, out_features=6*6)

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Un-Flatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(-1, 6, 6))

        ### shortcut: Center crop elements
        self.shorcut = nn.Sequential(transforms.CenterCrop(6),nn.Identity())  

    def forward(self, x):
        residual = self.shorcut(x)
        x = self.activation(self.conv1(x))
        x = self.conv2(x)
        # ## Shortcut addition
        x += residual

        # # COT
        x = self.flatten(x)

        # # CDER
        # y = self.flatten(x)

        x = self.fc2a(self.fc1a(x))
        # y = self.fc2b(self.fc1b(y))

        x = self.unflatten(x)
        # y = self.unflatten(y)
        # x = torch.cat((x, y),dim = 1)
        return x

if __name__=="__main__":
    model = DNN2w()
    batch_size = 16
    summary(model, input_size=(batch_size,2, 10, 10))
    # print(model)
