from torch.nn import CrossEntropyLoss
CEL = CrossEntropyLoss

# Automatically create a list of all classes imported in this file
import sys
import inspect
LOSSES = [name for name, obj in sys.modules[__name__].__dict__.items() if inspect.isclass(obj)]
print(f'{LOSSES =}')