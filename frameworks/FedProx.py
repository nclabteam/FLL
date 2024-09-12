import copy
import torch
from .base import Server, Client

optional = {
    'mu': 0.001,
}
compulsory = {
    'optimizer': 'PerturbedGradientDescent',
}

class FedProx(Server):
    pass

class FedProx_Client(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_params = copy.deepcopy(list(self.model.parameters()))

    def get_optimizer(self):
        """
        Returns the optimizer.
        """
        return getattr(
            __import__('optimizers'), 
            self.optimizer
        )(
            self.model.parameters(), 
            lr=self.learning_rate, 
            mu=self.mu
        )
    
    def _optim_step(self):
        """
        Performs an optimization step.
        """
        self.optimizer.step(
            global_params=self.global_params, 
            device=self.device
        )
    
    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                
                gm = torch.cat([p.data.view(-1) for p in self.global_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm-pm, p=2)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        loss = losses / train_num
        self.metrics['losses'].append(loss)
        return losses, train_num