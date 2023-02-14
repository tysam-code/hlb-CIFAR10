import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def make_net(data, scaling_factor, device, pad_amount, model_name):
    #model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model = torchvision.models.get_model(model_name, weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()

    return model