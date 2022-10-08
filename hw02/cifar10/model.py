import torch
import torchsummary
from torch import nn
from torchvision import models


def test_cifar10_model(num_classes=10):
    resnet_model = models.resnet18(pretrained=True)
    dim = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(dim, num_classes)

    return resnet_model


if __name__ == '__main__':
    m = test_cifar10_model()
    output = m(torch.ones((16, 3, 256, 256)))
    torchsummary.summary(m.cuda(), (3, 256, 256))
