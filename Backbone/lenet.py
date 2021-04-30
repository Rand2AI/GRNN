# -*-coding:utf-8-*-
import torch.nn as nn
import torch.nn.functional as func


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = func.sigmoid(self.conv1(x))
        x = func.max_pool2d(x, 2)
        x = func.sigmoid(self.conv2(x))
        x = func.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = func.sigmoid(self.fc1(x))
        x = func.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
