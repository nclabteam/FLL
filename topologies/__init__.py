from .base import BaseTopology
from .FullyConnected import FullyConnected
from .Ring import Ring

# Automatically create a list of all classes imported in this file
import sys
import inspect
TOPOLOGIES = [name for name, obj in sys.modules[__name__].__dict__.items() if inspect.isclass(obj)]
print(f'{TOPOLOGIES = }')