from .ResNet import (
    ResNet10, 
    ResNet18, 
    ResNet22, 
    ResNet34
)

from .ResNetDim import (
    ResNet18_2x,
    ResNet18_0_0625x,
    ResNet18_0_125x,
    ResNet18_0_25x,
    ResNet18_0_5x,
    
    ResNet10_0_5x,
    ResNet10_2x,

    ResNet34_0_5x,
    ResNet34_2x,
)

from .CNN import (
    FedAvgCNN
)

# Automatically create a list of all classes or functions imported in this file
import sys
import inspect
MODELS = [name for name, obj in sys.modules[__name__].__dict__.items() if inspect.isclass(obj) or inspect.isfunction(obj)]
print(f'{MODELS = }')