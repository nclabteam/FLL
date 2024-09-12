import time
import random
from .base import Server, Client

optional = {
    'ft_epochs': 10, 
    'ft_module': ['head', 'base'], 
    'save_local_model':True
}
compulsory = {
    'decoupling': True,
}

class FedBABU(Server):
    def train(self):
        super().train()
        s_t = time.time()
        for client in self.clients:
            client.fine_tune()
        self.metrics['time_per_iter'].append(time.time() - s_t)
        self.logger.info(f'Time cost: {self.metrics["time_per_iter"][-1]:.4f}s')
        self.logger.info('Evaluate fine-tuned personalized models')
        self.evaluate()
        self.save_results()
        self.save_models()

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(self.selected_clients, self.current_num_join_clients)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

class FedBABU_Client(Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for param in self.model.head.parameters():
            param.requires_grad = False
    
    def initialize_local(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()
    
    def fine_tune(self):
        self.logger.info('Fine-tuning the model')
        if 'head' in self.ft_module:
            for param in self.model.head.parameters():
                param.requires_grad = True
        if 'base' not in self.ft_module:
            for param in self.model.head.parameters():
                param.requires_grad = False
        self.epochs = self.ft_epochs
        self.train()
        