import torch
import torch.nn as nn


class VGGNet(nn.Module):
    def __init__(self, channels=1, num_classes=40):
        super(VGGNet, self).__init__()
        self.name = "VGGish"

        self.conv = nn.Sequential(
            # First convolution
            # Convolutional layer with kernel size (3x3), stride 1
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # First max pool
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Second convolution
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Second max pool
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Third convolution
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Third group but second convolution
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Third max pool
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Fourth group, first convolution
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Fourth group, second convolution
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # Third max pool
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = nn.Sequential(
            # Fully connected layer one
            nn.Dropout(0.5),
            nn.Linear(61440, 4096),
            nn.ReLU(),
            # Second fully connected layer
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            # Last fully connected layer
            nn.Dropout(0.5),
            nn.Linear(4096, 128),
        )
        
        # Extra layers for training (above gives a feature vector)
        self.training_layers = nn.Sequential(
            # First layer
            nn.Dropout(0.5),
            nn.Linear(128, 100),
            nn.ReLU(),
            # Second layer
            nn.Dropout(0.5),
            nn.Linear(100, num_classes),
        )

    # Runs classification with final output being a vector with each position representing a class
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.training_layers(x)
        return x

    # Returns a 128 dimensional embedding vector
    def inference(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
