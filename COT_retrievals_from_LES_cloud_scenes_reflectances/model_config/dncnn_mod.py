import torch.nn as nn
from torchinfo import summary
class DnCNNmod(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNNmod, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU())
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU())
        layers.append(nn.Flatten(start_dim=1))
        layers.append(nn.Linear(in_features=features*10*10, out_features=16*10*10))
        layers.append(nn.Linear(in_features=16*10*10, out_features=10*10))
        layers.append(nn.Unflatten(dim=1, unflattened_size=(-1, 10, 10)))
        # layers.append(nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        x = self.dncnn(x)
        return x

if __name__=="__main__":
    model = DnCNNmod(2,5)
    batch_size = 16
    summary(model, input_size=(batch_size,2, 10, 10))