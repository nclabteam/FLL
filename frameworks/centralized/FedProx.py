import copy
import torch
from .base import BaseServer, BaseClient

class FedProx_Server(BaseServer):
    pass

class FedProx_Client(BaseClient):
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

    def eval_on_train(self) -> tuple[float, int]:
        """
        Evaluates the local model on the training data.

        Returns:
            A tuple containing the average loss and number of samples.
        """
        trainloader = self.load_data(flag='train')
        self.model.eval()

        train_num: int = 0
        losses: float = 0
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

        return losses, train_num