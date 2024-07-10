import torch.nn as nn

class FedAvgCNN(nn.Module):
    """
    Source: https://github.com/TsingZ0/PFLlib/blob/master/system/flcore/trainmodel/models.py
    """
    def __init__(self, configs):
        dim=configs.dim
        in_features=configs.in_features
        num_classes=configs.num_classes

        super(FedAvgCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

# ==================================================================================================
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
# https://github.com/SamsungLabs/ordered_dropout/blob/master/od/layers/cnn.py

def od_conv_forward(layer, x, p=None):
    in_dim = x.size(1)  # second dimension is input dimension
    if p is None:  # i.e., don't apply OD
        out_dim = layer.out_channels
    else:
        if torch.is_tensor(p):
            p_scalar = p.cpu().item()
        else:
            p_scalar = p
        out_dim = int(np.ceil(layer.out_channels * p_scalar))
    # subsampled weights and bias
    weights_red = layer.weight[:out_dim, :in_dim]
    bias_red = layer.bias[:out_dim] if layer.bias is not None else None
    return layer._conv_forward(x, weights_red, bias_red)

class ODConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super(ODConv1d, self).__init__(*args, **kwargs)

    def forward(self, x, p=None):
        return od_conv_forward(self, x, p)

class ODConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(ODConv2d, self).__init__(*args, **kwargs)

    def forward(self, x, p=None):
        return od_conv_forward(self, x, p)

class ODConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super(ODConv3d, self).__init__(*args, **kwargs)

    def forward(self, x, p=None):
        return od_conv_forward(self, x, p)

class ODLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(ODLinear, self).__init__(*args, **kwargs)

    def forward(self, x, p=None):
        in_dim = x.size(1)  # second dimension is input dimension
        if p is None:  # i.e., don't apply OD
            out_dim = self.out_features
        else:
            if torch.is_tensor(p):
                p_scalar = p.cpu().item()
            else:
                p_scalar = p
            out_dim = int(np.ceil(self.out_features * p_scalar))
        # subsampled weights and bias
        weights_red = self.weight[:out_dim, :in_dim]
        bias_red = self.bias[:out_dim] if self.bias is not None else None
        return F.linear(x, weights_red, bias_red)

class FjORDCNN(nn.Module):
    def __init__(self, configs):
        dim=configs.dim
        in_features=configs.in_features
        num_classes=configs.num_classes

        super(FjORDCNN, self).__init__()

        self.conv1 = ODConv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True)
        self.conv2 = ODConv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True)
        self.fc1 = ODLinear(dim, 512)
        self.fc2 = ODLinear(512, num_classes)

    def forward(self, x):
        x, p = x
        b = x.shape[0]
        x = F.relu(F.max_pool2d(self.conv1(x, p=p), 2))
        x = F.relu(F.max_pool2d(self.conv2(x, p=p), 2))
        x = x.view(b, -1)
        x = F.relu(self.fc1(x, p=p))
        x = self.fc2(x, p=1.) 
        return x
    
