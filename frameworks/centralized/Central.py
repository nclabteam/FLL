from .base import Server, Client
from torch.utils.data import ConcatDataset

class Central(Server):
    def server_aggregation(self):
        pass

    def receive_models(self):
        pass
    
    def get_loss(self):
        if type(self.loss) != str: return
        self.loss = getattr(__import__('losses'), self.loss)()

    def get_optimizer(self):
        if type(self.optimizer) != str: return
        self.optimizer = getattr(__import__('optimizers'), self.optimizer)(self.model.parameters(), lr=self.learning_rate)
    
    def _optim_step(self):
        self.optimizer.step()

    def train_clients(self):
        self.get_loss()
        self.get_optimizer()
        self.global_model.train()
        for _ in range(self.epochs):
            for client in self.selected_clients:
                trainloader = client.load_train_data()
                for x, y in trainloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = self.global_model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self._optim_step() 
                client.fix_results()

class Central_Client(Client):
    pass
