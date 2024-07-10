import copy
import torch
import numpy as np
import torch.nn as nn
from .base import BaseServer, BaseClient

class FedCAC_Server(BaseServer):
    def send_models(self):
        if self.current_iter != 0:
            self.get_customized_global_models()
        super().send_models()
    
    def get_customized_global_models(self):
        """
        Overview:
            Aggregating customized global models for clients to collaborate critical parameters.
        """
        assert isinstance(self.beta, int) and self.beta >= 1
        overlap_buffer = self.calculate_overlap_buffer()

        # Calculate the global threshold
        overlap_buffer_tensor = torch.tensor(overlap_buffer)
        overlap_sum = overlap_buffer_tensor.sum()
        overlap_avg = overlap_sum / ((self.num_clients - 1) * self.num_clients)
        overlap_max = overlap_buffer_tensor.max()
        threshold = overlap_avg + (self.current_iter + 1) / self.beta * (overlap_max - overlap_avg)

        # Calculate the customized global model for each client
        for i in range(self.num_clients):
            self.calculate_customized_model(i, overlap_buffer[i], threshold)

    def calculate_overlap_buffer(self):
        """
        Calculate overlap rate between clients.
        """
        overlap_buffer = [[] for _ in range(self.num_clients)]
        for i in range(self.num_clients):
            for j in range(self.num_clients):
                if i == j:
                    continue
                overlap_rate = 1 - torch.sum(
                    torch.abs(self.clients[i].critical_parameter.to(self.device) - self.clients[j].critical_parameter.to(self.device))
                ) / (2 * torch.sum(self.clients[i].critical_parameter.to(self.device)).cpu().item())
                overlap_buffer[i].append(overlap_rate)
        return overlap_buffer

    def calculate_customized_model(self, client_index, overlap_rates, threshold):
        """
        Calculate and set the customized global model for a specific client.
        """
        w_customized_global = copy.deepcopy(self.clients[client_index].model.state_dict())
        collaboration_clients = [client_index]

        # Find clients whose critical parameter locations are similar to the given client
        for idx, overlap_rate in enumerate(overlap_rates):
            if overlap_rate >= threshold:
                collaboration_clients.append(idx)

        for key in w_customized_global.keys():
            for client in collaboration_clients:
                if client == client_index:
                    continue
                w_customized_global[key] += self.clients[client].model.state_dict()[key]
            w_customized_global[key] = torch.div(w_customized_global[key], float(len(collaboration_clients)))

        # Send the customized global model to the client
        self.clients[client_index].customized_model.load_state_dict(w_customized_global)


class FedCAC_Client(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critical_parameter = None
        self.customized_model = copy.deepcopy(self.model)
        self.global_mask = None
        self.local_mask = None
    
    def initialize_local(self, model: nn.Module):
        """
        Initialize the local model parameters.
        """
        if self.local_mask is not None:
            self.apply_masks(model)
        else:
            super().initialize_local(model)
    
    def apply_masks(self, model: nn.Module):
        """
        Apply local and global masks to initialize model parameters.
        """
        for index, ((_, param1), (_, param2), (_, param3)) in enumerate(
                zip(self.model.named_parameters(), model.named_parameters(), self.customized_model.named_parameters())):
            param1.data = self.local_mask[index].to(self.device).float() * param3.data + \
                          self.global_mask[index].to(self.device).float() * param2.data
    
    def train(self):
        """
        Trains the local model on the training data.
        """
        initial_model = copy.deepcopy(self.model)
        super().train()
        self.critical_parameter, self.global_mask, self.local_mask = self.evaluate_critical_parameter(
            prevModel=initial_model, model=self.model, tau=self.tau
        )
    
    def evaluate_critical_parameter(self, prevModel: nn.Module, model: nn.Module, tau: float):
        """
        Overview:
            Implement critical parameter selection.
        """
        global_mask, local_mask, critical_parameter = [], [], []

        for (name1, prevparam), (name2, param) in zip(prevModel.named_parameters(), model.named_parameters()):
            mask, global_m, local_m = self.compute_masks(prevparam, param, tau)
            global_mask.append(global_m)
            local_mask.append(local_m)
            critical_parameter.append(mask)

        model.zero_grad()
        critical_parameter = torch.cat(critical_parameter)

        return critical_parameter, global_mask, local_mask

    def compute_masks(self, prevparam, param, tau):
        """
        Compute masks to identify critical parameters.
        """
        g = param.data - prevparam.data
        c = torch.abs(g * param.data)
        metric = c.view(-1)
        nz = int(tau * metric.size(0))
        top_values, _ = torch.topk(metric, nz)
        thresh = top_values[-1] if top_values.numel() > 0 else np.inf

        if thresh <= 1e-10:
            thresh = self.adjust_threshold(metric)

        mask = (c >= thresh).int().to('cpu')
        global_mask = (c < thresh).int().to('cpu')
        local_mask = mask

        return mask.view(-1), global_mask, local_mask

    def adjust_threshold(self, metric):
        """
        Adjust threshold to handle zero values.
        """
        new_metric = metric[metric > 1e-20]
        if new_metric.numel() == 0:
            print(f'Abnormal!!! metric:{metric}')
            return np.inf
        return new_metric.sort()[0][0]
