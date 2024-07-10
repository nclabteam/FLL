from torch.optim import Adam, SGD
from .PGD import PerturbedGradientDescent

from .SPS import StochasticPolyakStepsize
SPS = StochasticPolyakStepsize

from .SLS import StochasticLineSearch
SLS = StochasticLineSearch

from .AdaSLS import AdaSLS