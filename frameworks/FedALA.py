from .base import Server, Client

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

# ==============================

import copy
import torch
import random
import numpy as np
import torch.nn as nn
from typing import List, Tuple
from torch.utils.data import DataLoader

class ALA:
    def __init__(
        self,
        configs,
        loss: nn.Module,
        train_data: List[Tuple], 
        logger=None,
    ) -> None:
        self.loss = loss
        self.train_data = train_data
        self.batch_size = configs.batch_size
        self.logger = logger
        self.rand_percent = configs.data_rand_percent
        self.layer_idx = configs.p
        self.eta = configs.eta
        self.threshold = configs.threshold
        self.num_pre_loss = configs.local_patience
        self.device = configs.device

        self.weights = None # Learnable local aggregation weights.
        self.start_phase = True


    def adaptive_local_aggregation(
        self, 
        global_model: nn.Module,
        local_model: nn.Module
    ) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. 

        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 

        Returns:
            None.
        """

        # randomly sample partial local training data
        rand_num = int(self.rand_percent*len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data)-rand_num)
        dataset = self.train_data.dataset  # Extract the dataset from the DataLoader
        subset_data = dataset[rand_idx:rand_idx+rand_num]  # Slice the dataset
        rand_loader = DataLoader(subset_data, batch_size=self.batch_size, drop_last=False)

        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()


        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]
        params_gp = params_g[-self.layer_idx:]
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                self.weights):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y) # modify according to the local objective
                loss_value.backward()

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                self.logger.info(f'ALA epochs: {str(cnt).zfill(3)} | Std: {np.std(losses[-self.num_pre_loss:])}')
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()