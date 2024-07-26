from torchvision.models import resnet18, resnet34
from torchvision.models.resnet import ResNet, BasicBlock

def ResNet10(configs):
    # 10 = 2 + 2 * (1 + 1 + 1 + 1)
    return ResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=configs.num_classes)

def ResNet18(configs):
    return resnet18(
        num_classes=configs.num_classes
    ).to(configs.device)

def ResNet22(configs):
    # 22 = 2 + 2 * (2 + 3 + 3 + 2)
    return ResNet(block=BasicBlock, layers=[2, 3, 3, 2], num_classes=configs.num_classes)

def ResNet34(configs):
    return resnet34(
        num_classes=configs.num_classes
    ).to(configs.device)