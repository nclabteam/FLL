import os
import copy
import time
import torch
import random
import logging
import threading
import numpy as np
import polars as pl
import torch.nn as nn
from argparse import Namespace
from torch.utils.data import DataLoader
from utils import (
    ModelSummary,
    zero_parameters,
    plot_accuracy_granularity,
    plot_participant_rate
)

class SharedMethods:
    default_value = -1.0

    def save_results(self):
        pl_df = pl.DataFrame(self.metrics)
        path = os.path.join(self.result_path, self.name.lower().strip()+'.csv')
        pl_df.write_csv(path)
        self.logger.info(f'Results saved to {path}')
    
    def save_model(self):
        path = os.path.join(self.model_path, self.name.lower().strip()+'.pt')
        torch.save(self.model, path)
        self.logger.info(f'Model saved to {path}')
    
    def fix_results(self, default=-1.0):
        max_length = max(len(lst) for lst in self.metrics.values())
        for key in self.metrics.keys():
            if len(self.metrics[key]) < max_length:
                self.metrics[key].extend([default] * (max_length - len(self.metrics[key])))
    
    def load_data(self, path):
        with open(path, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()

        x = torch.Tensor(data['x']).type(torch.float32)
        y = torch.Tensor(data['y']).type(torch.int64)

        return [(x, y) for x, y in zip(x, y)]
    
    def make_logger(self, name, path):
        log_path = os.path.join(path, f'{name.lower().strip()}.log')
        
        # Create a unique logger name using the instance id
        logger_name = f'{name}_{self.times}'
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Create file and stream handlers
        file_handler = logging.FileHandler(log_path)
        stream_handler = logging.StreamHandler()
        
        # Set logging format
        formatter = logging.Formatter(f'%(asctime)s ~ %(levelname)s ~ %(lineno)-4.4d ~ {name} ~ %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        
        self.logger.info(f'Logger created at {log_path}')
    
    def set_configs(self, configs, **kwargs):
        if isinstance(configs, Namespace):
            for key, value in vars(configs).items():
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.configs = configs
    
    def mkdir(self):
        self.save_path = os.path.join(self.save_path, str(self.times))
        self.model_path = os.path.join(self.save_path, 'models')
        self.model_info_path = os.path.join(self.save_path, 'models_info')
        self.log_path = os.path.join(self.save_path, 'logs')
        self.result_path = os.path.join(self.save_path, 'results')
        for dir in [
            self.save_path,
            self.model_path,
            self.log_path,
            self.model_info_path,
            self.result_path,
        ]:
            if not os.path.exists(dir):
                os.makedirs(dir)

class Server(SharedMethods):
    def __init__(self, configs, times):
        self.set_configs(configs=configs, times=times)
        self.mkdir()

        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients

        self.name = '  SERVER  '

        self.metrics = {
            'test_personal_accs': [],
            'test_global_accs': [],
            'losses': [],
            'time_per_iter': [],
        }
        self.test_file = os.path.join(self.dataset_path, 'test', 'server.npz')

    def get_model(self):
        self.global_model = getattr(__import__('models'), self.model)(configs=self.configs).to(self.device)
        if self.decoupling:
            head = copy.deepcopy(self.global_model.fc)
            self.global_model.fc = nn.Identity()
            self.global_model = getattr(__import__('models'), 'BaseHeadSplit')(self.global_model, head).to(self.device)

    def set_clients(self):
        module_name = self.__module__
        class_name = self.__class__.__name__ + '_Client'
        client_object = getattr(__import__(module_name, fromlist=[class_name]), class_name)
        self.clients = [client_object(self.configs, id, self.global_model, self.times) for id in range(self.num_clients)]

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        self.selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

    def models_sumary(self):
        testloader = self.clients[0].load_test_data(shuffle=False)
        ModelSummary(
            model=self.global_model, 
            save_path=os.path.join(self.model_info_path, 'server_model.svg'),
            dataloader=testloader
        )()

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            s = time.time()
            client.initialize_local(self.global_model)
            client.metrics['send_time'].append(2 * (time.time() - s))

    def receive_data_from_clients(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(self.selected_clients, self.current_num_join_clients)

        self.uploaded_weights = []
        self.uploaded_models = []
        for client in active_clients:
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)

    def server_aggregation(self):
        assert (len(self.uploaded_models) > 0)

        self.uploaded_weights = np.array(self.uploaded_weights) / np.sum(self.uploaded_weights)

        self.global_model = zero_parameters(self.uploaded_models[0])
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

    def save_model(self):
        metric = self.metrics['test_personal_accs'] if self.save_local_model else self.metrics['test_global_accs']
        if metric[-1] == max(metric):
            super().save_model()
            if not self.save_local_model: return    
            for client in self.clients:
                client.save_model()

    def save_results(self):
        super().save_results()
        for client in self.clients:
            client.save_results()
        
        # Save the indices where accuracies reach every 5% increment
        granularity = 5
        granularity_df = {
            "server_personal": self.get_granularity_indices(self.metrics["test_personal_accs"], granularity=5),
            "server_general": self.get_granularity_indices(self.metrics["test_global_accs"], granularity=5),
        }
        for client in self.clients:
            granularity_df[f'client_{client.id}'] = self.get_granularity_indices(client.metrics['accs'], granularity=5)

        granularity_df = pl.DataFrame(granularity_df).transpose(
            include_header=True, 
            column_names=[str(number) for number in range(0, 101, granularity)][1:],
            header_name='accuracy'
        )
        granularity_path = os.path.join(self.result_path, f'accuracy_granularity.csv')
        granularity_df.write_csv(granularity_path)
        self.logger.info(f'Accuracy granularity results saved to {granularity_path}')

        plot_accuracy_granularity(
            data=pl.read_csv(granularity_path, has_header=False), 
            figsize=(14, 8*max(1, self.num_clients//20)),
            save_path=os.path.join(self.save_path, 'heatmap.png')
        )
        plot_participant_rate(
            data = {client.id: int(len(client.metrics['train_time'])-client.metrics['train_time'].count(-1.0)) for client in self.clients},
            figsize=(10*max(1, self.num_clients//20-1), 5),
            save_path=os.path.join(self.save_path, 'participant.png')
        )
    
    def get_granularity_indices(self, accuracies, granularity=5):
        """
        Returns the indices (epochs) where the accuracies reach each granularity level.
        """
        granularity_levels = list(range(0, 101, granularity))[1:]
        granularity_indices = []

        for level in granularity_levels:
            for idx, acc in enumerate(accuracies):
                if acc*100 >= level:
                    granularity_indices.append(idx)
                    break
            else:
                granularity_indices.append(None)
        return granularity_indices

    def test_metrics(self):       
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_metrics()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        accuracy = sum(tot_correct) / sum(num_samples)
        std = np.std([a / n for a, n in zip(tot_correct, num_samples)])
        self.metrics["test_personal_accs"].append(accuracy)
        self.logger.info(f'Personal Test Accurancy: {accuracy*100:.2f}±{std:.2f}%')

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)
        train_loss = sum(losses) / sum(num_samples)
        self.metrics["losses"].append(train_loss)
        self.logger.info(f'Averaged Train Loss: {train_loss:.4f}')

    def global_test_metrics(self):
        num_samples = []
        tot_correct = []
        testloader = DataLoader(self.load_data(self.test_file), self.batch_size, drop_last=False, shuffle=False)
        self.global_model.eval()
        for x, y in testloader:
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.global_model(x)
            _, pred = output.max(1)
            correct = pred.eq(y)
            tot_correct.append(correct.sum().item())
            num_samples.append(y.size(0))
        accuracy = sum(tot_correct) / sum(num_samples)
        std = np.std([a / n for a, n in zip(tot_correct, num_samples)])
        self.metrics["test_global_accs"].append(accuracy)
        self.logger.info(f'Global Test Accuracy: {accuracy*100:.2f}±{std:.2f}%')

    def evaluate(self):
        if self.current_iter%self.eval_gap == 0:
            self.logger.info('')
            self.logger.info(f'-------------Round number: {str(self.current_iter).zfill(4)}-------------')
            self.train_metrics()
            self.test_metrics()
            self.global_test_metrics()

    def early_stopping(self, acc_lss: list[list[float]], div_value: float = None) -> bool:
        if self.patience is None: return False
        for acc_ls in acc_lss:
            if len(acc_ls) < 2: return False

            # Determine the index of the highest accuracy
            top_index = torch.topk(torch.tensor(acc_ls), 1).indices[0].item()
            # Get the most recent accuracy values, up to the number of patience steps
            recent_accuracies = acc_ls[-self.patience:]

            # Check if the highest accuracy was achieved more than 'patience' epochs ago
            self.logger.info(f'Patience: {(len(acc_ls) - top_index)}')
            patience_condition = (len(acc_ls) - top_index) > self.patience
            # Check if the standard deviation of the recent accuracies is below 'div_value'
            std_condition = np.std(recent_accuracies) < div_value if div_value is not None else True
            
            # Convergence requires both conditions to be met, if both are specified
            if patience_condition and std_condition:
                self.logger.info('Early stopping condition met.')
                return True

        return False
    
    def train_clients(self):
        # Divide clients into batches as evenly as possible
        client_batches = []
        for i in range(self.workers):
            client_batches.append([])
            for j, client in enumerate(self.selected_clients):
                if j % self.workers == i:
                    client_batches[i].append(client)

        threads = []
        for batch in client_batches:
            # Create a thread for each batch
            thread = threading.Thread(target=self._train_batch, args=(batch,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()
        
        for client in self.clients:
            client.fix_results(self.default_value)

    def _train_batch(self, clients):
        """Trains a batch of clients sequentially within a thread."""
        for client in clients:
            client.train()

    def train(self):
        self.make_logger(name=self.name, path=self.log_path)
        self.get_model()
        self.set_clients()
        self.models_sumary()
        for i in range(self.iterations+1):
            s_t = time.time()
            self.current_iter = i
            self.select_clients()
            self.send_models()
            self.evaluate()
            self.train_clients()
            self.receive_data_from_clients()
            self.server_aggregation()
            self.metrics['time_per_iter'].append(time.time() - s_t)
            self.logger.info(f'Time cost: {self.metrics["time_per_iter"][-1]:.4f}s')
            self.save_model()
            if self.early_stopping(acc_lss=[self.metrics["test_global_accs"]]): 
                break

        self.logger.info('')
        self.logger.info('-'*50)
        self.save_results()
        self.logger.info('-'*50)
        self.logger.info('')
        max_key_length = max(len(key) for key in self.metrics.keys())
        header = f"{'Metric':<{max_key_length}} | {'Min':>8} | {'Mean':>8} | {'Max':>8}"
        self.logger.info(header)
        self.logger.info('-'*len(header))
        for key, value in self.metrics.items():
            value = np.array(value)
            self.logger.info(f'{key:<{max_key_length}} | {value.min():8.4f} | {value.mean():8.4f} | {value.max():8.4f}')

class Client(SharedMethods):
    def __init__(self, configs: dict, id: int, model: nn.Module, times: int):
        self.set_configs(configs=configs, id=id, times=times) 
        self.mkdir()
        self.model = copy.deepcopy(model)

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.metrics = {
            'train_time': [],
            'send_time': [],
            'accs': [],
            'losses': [],
        }

        self.get_loss()
        self.get_optimizer()

        self.train_file = os.path.join(self.dataset_path, 'train/', str(self.id) + '.npz')
        self.test_file = os.path.join(self.dataset_path, 'test/', str(self.id) + '.npz')

        self.name = f'CLIENT_{str(self.id).zfill(3)}'
        self.make_logger(name=self.name, path=self.log_path)

    def get_loss(self):
        self.loss = getattr(__import__('losses'), self.loss)()

    def get_optimizer(self):
        optimizer_class = getattr(__import__('optimizers'), self.optimizer)
        optimizer_params = {'lr': self.learning_rate}
        
        if self.optimizer.lower() == 'sgd' and hasattr(self, 'momentum'):
            optimizer_params['momentum'] = self.momentum
        
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        
    def load_train_data(self, shuffle=True):
        return DataLoader(self.load_data(self.train_file), self.batch_size, drop_last=True, shuffle=shuffle)

    def load_test_data(self, shuffle=True):
        return DataLoader(self.load_data(self.test_file), self.batch_size, drop_last=False, shuffle=shuffle)
        
    def initialize_local(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def test_metrics(self):
        testloader = self.load_test_data()
        self.test_samples = len(testloader)
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in testloader:
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
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        loss = losses / train_num
        self.metrics['losses'].append(loss)
        return losses, train_num
    
    def _optim_step(self):
        self.optimizer.step()

    def train(self):
        trainloader = self.load_train_data()
        self.train_samples = len(trainloader)
        self.model.train()
        start_time = time.time()
        for _ in range(self.epochs):
            for x, y in trainloader:
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self._optim_step() 

        self.metrics['train_time'].append(time.time() - start_time)