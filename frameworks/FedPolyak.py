from .base import Server, Client
from utils import Polyak

class FedPolyak(Server):
    pass

class FedPolyak_Client(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polyak = Polyak(mix=self.mix, model=self.model)
    
    def train(self):
        super().train()
        self.polyak.add_weight(model=self.model)
    
    def initialize_local(self, model):
        if self.polyak.num_weights == 0:
            super().initialize_local(model)
        else:
            self.model = self.polyak.models_mixture(old=self.model, new=model)