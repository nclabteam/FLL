from torch.optim import Adam, SGD
from .FedProx import PerturbedGradientDescent

# Automatically create a list of all classes imported in this file
import sys
import inspect
OPTIMIZERS = [name for name, obj in sys.modules[__name__].__dict__.items() if inspect.isclass(obj)]
print(f'{OPTIMIZERS = }')