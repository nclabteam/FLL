import copy
import torch
import numpy as np
from scipy import linalg
import torch.nn.functional as F
from .base import Server, Client

optional = {
    'epsilon':1.2, 
    'ord':2, 
    'dp':0.001
}

class FedAtt(Server):
    def server_aggregation(self):
        """
        Source: https://github.com/shaoxiongji/fed-att/blob/master/src/agg/aggregate.py
        """
        assert (len(self.uploaded_models) > 0)

        w_server = self.global_model.state_dict()
        w_clients = [m.state_dict() for m in self.uploaded_models]

        w_next = copy.deepcopy(w_server)
        att = {}
        for k in w_server.keys():
            w_next[k] = torch.zeros_like(w_server[k]).cpu()
            att[k] = torch.zeros(len(w_clients)).cpu()
        
        for k in w_next.keys():
            w_server_flat = w_server[k].cpu().numpy().flatten()
            for i in range(len(w_clients)):
                w_client_flat = w_clients[i][k].cpu().numpy().flatten()
                att[k][i] = torch.from_numpy(np.array(linalg.norm(w_server_flat - w_client_flat, ord=self.ord)))
        
        for k in w_next.keys():
            att[k] = F.softmax(att[k], dim=0)
        
        for k in w_next.keys():
            att_weight = torch.zeros_like(w_server[k])
            for i in range(len(w_clients)):
                att_weight += torch.mul(w_server[k] - w_clients[i][k], att[k][i])
            w_next[k] = w_server[k] - torch.mul(att_weight, self.epsilon) + torch.mul(torch.randn(w_server[k].shape, device=self.device), self.dp)
        
        self.global_model.load_state_dict(w_next)

class FedAtt_Client(Client):
    pass
