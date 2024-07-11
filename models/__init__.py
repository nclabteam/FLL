from .CNN import (
    FedAvgCNN,
    FjORDCNN
)

from .ResNet import (
    ResNet10, 
    ResNet18, 
    ResNet22, 
    ResNet34
)

from .ResNetDim import (
    ResNet10Half,
    ResNet18Half,
    ResNet34Half,
    ResNet10Double,
    ResNet18Double,
    ResNet34Double,
)

# Automatically create a list of all classes or functions imported in this file
import sys
import inspect
MODELS = [name for name, obj in sys.modules[__name__].__dict__.items() if inspect.isclass(obj) or inspect.isfunction(obj)]
print(f'{MODELS =}')