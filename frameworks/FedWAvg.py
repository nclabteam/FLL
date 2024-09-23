import torch
import random
import numpy as np
from .base import Server, Client

class FedWAvg(Server):
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
            self.uploaded_weights.append(client.forgettable_example_count)
            self.uploaded_models.append(client.model)
        alpha = 0.3
        self.uploaded_weights = np.array(self.uploaded_weights) / (np.sum(self.uploaded_weights) + 1e-6)
        self.uploaded_weights = (1 - alpha) + alpha * (self.uploaded_weights * len(self.uploaded_models))
        self.uploaded_weights = np.array(self.uploaded_weights) / np.sum(self.uploaded_weights)

class FedWAvg_Client(Client):
    def initialize_local(self, model):
        trainloader = self.load_train_data()
        self.model.eval()
        model.eval()
        self.forgettable_example_count = 0

        with torch.no_grad():
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                pred = torch.argmax(self.model(x), dim=1)
                pred_global = torch.argmax(model(x), dim=1)
                
                self.forgettable_example_count += torch.sum((pred == y) & (pred_global != y)).item()

        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()
