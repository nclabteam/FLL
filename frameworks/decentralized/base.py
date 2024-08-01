import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from frameworks.base import SharedMethods
from utils import (
    ModelSummary,
    zero_parameters
)

class TrainingController(SharedMethods):
    def __init__(self, configs, times):
        self.set_configs(configs=configs, times=times)
        self.mkdir()
        self.metrics = {
            'test_personal_accs': [],
            'losses': [],
            'time_per_iter': [],
        }
    
    def get_client_object(self):
        return getattr(__import__('frameworks'), self.framework+'_Node')
    
    def set_nodes(self, nodeObj=None):
        if nodeObj is None: nodeObj = self.get_client_object()
        self.nodes = []
        for id, neighbor in self.neighbors.items():
            self.nodes.append(nodeObj(id=id, configs=self.configs, times=self.times, neighbors=neighbor))

    def set_neighbors(self):
        self.neighbors = getattr(__import__('topologies'), self.topology)(self.num_clients).connections

    def test_metrics(self):       
        num_samples = []
        tot_correct = []
        for node in self.nodes:
            ct, ns = node.test_metrics()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        accuracy = sum(tot_correct) / sum(num_samples)
        std = np.std([a / n for a, n in zip(tot_correct, num_samples)])
        self.metrics["test_personal_accs"].append(accuracy)
        self.logger.info(f'Personal Test Accurancy: {accuracy*100:.2f}±{std:.2f}%')

    def train_metrics(self):
        num_samples = []
        losses = []
        for node in self.nodes:
            cl, ns = node.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
        train_loss = sum(losses) / sum(num_samples)
        self.metrics["losses"].append(train_loss)
        self.logger.info(f'Averaged Train Loss: {train_loss:.4f}')

    def evaluate(self):
        if self.current_iter%self.eval_gap == 0:
            self.logger.info('')
            self.logger.info(f'-------------Round number: {str(self.current_iter).zfill(4)}-------------')
            self.train_metrics()
            self.test_metrics()

    def train_nodes(self):
        for node in self.nodes:
            node.train()

    def send_models(self):
        for node in self.nodes:
            node.uploaded_models = [self.nodes[nei].model for nei in node.neighbors]
            node.uploaded_weights = [self.nodes[nei].train_samples for nei in node.neighbors]
            node.uploaded_models.append(node.model)
            node.uploaded_weights.append(node.train_samples)
            tot_samples = sum(node.uploaded_weights)
            for i, w in enumerate(node.uploaded_weights):
                node.uploaded_weights[i] = w / tot_samples

    def nodes_aggregation(self):
        for node in self.nodes:
            node.aggregate()

    def save_models(self):
        if self.metrics["test_personal_accs"][-1] == max(self.metrics["test_personal_accs"]):
            for node in self.nodes:
                node.save_model()

    def train(self):
        self.make_logger(name='TMANAGER', path=self.log_path)
        self.set_neighbors()
        self.set_nodes()
        for i in range(self.iterations+1):
            s_t = time.time()
            self.current_iter = i
            self.evaluate()
            self.train_nodes()
            self.send_models()
            self.nodes_aggregation()
            self.save_models()
            self.metrics['time_per_iter'].append(time.time() - s_t)
            self.logger.info(f'Time cost: {self.metrics["time_per_iter"][-1]:.4f}s')

class Node(SharedMethods):
    def __init__(self, id, configs, times, neighbors):
        self.set_configs(configs=configs, times=times, id=id, neighbors=neighbors)
        self.mkdir()
        self.metrics = {
            'train_time': [],
            'send_time': [],
            'accs': [],
            'losses': [],
        }
        self.get_model()
        self.get_loss()
        self.get_optimizer()
        self.train_file = os.path.join(self.dataset_path, 'train/', str(self.id) + '.npz')
        self.test_file = os.path.join(self.dataset_path, 'test/', str(self.id) + '.npz')
        self.make_logger(name=f'NODE_{str(self.id).zfill(3)}', path=self.log_path)
    
    def get_loss(self):
        self.loss = getattr(__import__('losses'), self.loss)()

    def get_optimizer(self):
        self.optimizer = getattr(__import__('optimizers'), self.optimizer)(self.model.parameters(), lr=self.learning_rate)
    
    def get_model(self):
        self.model = getattr(__import__('models'), self.model)(configs=self.configs).to(self.device)
    
    def load_data(self, path):
        with open(path, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()

        x = torch.Tensor(data['x']).type(torch.float32)
        y = torch.Tensor(data['y']).type(torch.int64)

        return [(x, y) for x, y in zip(x, y)]
        
    def load_train_data(self, shuffle=True):
        return DataLoader(self.load_data(self.train_file), self.batch_size, drop_last=True, shuffle=shuffle)

    def load_test_data(self, shuffle=True):
        return DataLoader(self.load_data(self.test_file), self.batch_size, drop_last=False, shuffle=shuffle)

    def train(self):
        trainloader = self.load_train_data()
        self.train_samples = len(trainloader)
        self.model.train()
        start_time = time.time()
        for _ in range(self.epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self._optim_step() 

        self.metrics['train_time'].append(time.time() - start_time)
    
    def aggregate(self):
        assert len(self.uploaded_models) == len(self.neighbors)+1
        self.model = zero_parameters(self.model)
        for w, neighbor in zip(self.uploaded_weights, self.uploaded_models):
            for model_param, neightbor_param in zip(self.model.parameters(), neighbor.parameters()):
                model_param.data += w * neightbor_param.data.clone()
    
    def _optim_step(self):
        self.optimizer.step()

    def save_model(self):
        path = os.path.join(self.model_path, f'node_{str(self.id).zfill(3)}.pt')
        torch.save(self.model, path)
        self.logger.info(f'Model saved to {path}')
    
    def test_metrics(self):
        testloader = self.load_test_data()
        self.test_samples = len(testloader)
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
        acc = test_acc/test_num
        self.metrics['accs'].append(acc)
        return test_acc, test_num

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
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        loss = losses / train_num
        self.metrics['losses'].append(loss)
        return losses, train_num