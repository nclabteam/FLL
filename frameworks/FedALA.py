from .base import Server, Client
from utils import ALA

class FedALA(Server):
    pass

class FedALA_Client(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        train_data = self.load_train_data()
        self.ALA = ALA(
            configs=self.configs,
            loss=self.loss, 
            train_data=train_data, 
            logger=self.logger
        )
        
    def initialize_local(self, model):
        self.ALA.adaptive_local_aggregation(model, self.model)