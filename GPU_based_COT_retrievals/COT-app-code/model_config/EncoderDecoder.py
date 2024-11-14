import torch
import torch.nn as nn
from torchinfo import summary

class EncoderDecoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=1,
            padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=1,
            padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=1, 
            padding=1, output_padding=0)
        )

### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(10 * 10 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 10 * 10 * 32),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        self.unflatten = nn.Unflatten(dim=1,unflattened_size=(32, 10, 10))

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        # x = torch.sigmoid(x)
        return x

if __name__=="__main__":
    model = EncoderDecoder(encoded_space_dim=4)
    batch_size = 16
    summary(model, input_size=(batch_size, 2, 10, 10))
    print("It works")