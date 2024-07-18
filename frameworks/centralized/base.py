import torch
import os
import numpy as np
import copy
import time
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import ModelSummary
import polars as pl
import threading
from frameworks.base import SharedMethods

class BaseServer(SharedMethods):
    """
    Base server class for federated learning.
    """
    def __init__(self, configs: dict, times: int):
        """
        Initializes the server.

        configs:
            configs: A dictionary of arguments.
            times: The current time.
        """
        self.set_configs(configs=configs, times=times)
        self.mkdir()

        self.clients: list[BaseClient] = []
        self.selected_clients: list[BaseClient] = []
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients

        self.uploaded_weights: list[float] = []
        self.uploaded_ids: list[int] = []
        self.uploaded_models: list[nn.Module] = []

        self.metrics = {
            "test_personal_accs": [],
            "test_traditional_accs": [],
            "losses": [],
            "time_per_iter": [],
        }
    
    def get_model(self):
        """
        Returns the global model.
        """
        self.global_model = getattr(__import__('models'), self.model)(configs=self.configs).to(self.device)

    def get_client_object(self):
        return getattr(
                __import__(
                    self.__module__.replace(
                        'Server', 
                        'Client'
                    )
                ), 
                self.__class__.__name__.replace(
                    'Server', 
                    'Client'
                )
            )
    
    def set_clients(self, clientObj=None):
        """
        Sets the clients for the server.

        Args:
            clientObj: The client class. If None, dynamically determine the client class based on the server class name.
        """
        self.logger.info('Setting clients.')

        if clientObj is None: clientObj = self.get_client_object()

        for idx in range(self.num_clients):
            client = clientObj(self.configs, id=idx, model=self.global_model, times=self.times)
            self.clients.append(client)

        self.logger.info('Finished setting clients.')

    def select_clients(self):
        """
        Selects a random subset of clients to participate in the current round.
        """
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        self.selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

    def send_models(self):
        """
        Sends the global model to all clients.
        """
        assert (len(self.clients) > 0)

        for client in self.clients:
            s = time.time()
            client.initialize_local(self.global_model)
            client.metrics['send_time'].append(2 * (time.time() - s))

    def receive_models(self):
        """
        Receives the updated models from selected clients.
        """
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
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    
    def server_aggregation(self):
        """
        Performs model aggregation based on weighted average.
        """
        assert (len(self.uploaded_models) > 0)

        self.global_model = self.zero_model(self.global_model)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data * w

    def save_models(self):
        """
        Saves the current global model.
        """
        if self.metrics["test_personal_accs"][-1] == max(self.metrics["test_personal_accs"]):
            path = os.path.join(self.model_path, f'server.pt')
            torch.save(
                self.global_model, 
                path
            )
            self.logger.info(f'Model saved to {path}')
            for client in self.clients:
                client.save_model()

    def save_results(self):
        """
        Saves accuracy, loss, and AUC results as a single Polars DataFrame.
        """
        pl_df = pl.DataFrame(self.metrics)
        path = os.path.join(self.result_path, f'server.csv')
        pl_df.write_csv(path)
        self.logger.info(f'Results saved to {path}')

        for client in self.clients:
            client.save_results()
        
        # Save the indices where accuracies reach every 5% increment
        granularity = 5
        granularity_df = {
            "server_personal": self.get_granularity_indices(self.metrics["test_personal_accs"], granularity=5),
            "server_general": self.get_granularity_indices(self.metrics["test_traditional_accs"], granularity=5),
        }
        for client in self.clients:
            granularity_df[f'client_{client.id}'] = self.get_granularity_indices(client.metrics['accs'], granularity=5)

        granularity_df = pl.DataFrame(granularity_df).transpose(
            include_header=True, 
            column_names=[str(number) for number in range(0, 101, granularity)],
            header_name='accuracy'
        )
        granularity_path = os.path.join(self.result_path, f'accuracy_granularity.csv')
        granularity_df.write_csv(granularity_path)
        self.logger.info(f'Accuracy granularity results saved to {granularity_path}')

    def get_granularity_indices(self, accuracies, granularity=5):
        """
        Returns the indices (epochs) where the accuracies reach each granularity level.
        """
        granularity_levels = list(range(0, 101, granularity))
        granularity_indices = []

        for level in granularity_levels:
            for idx, acc in enumerate(accuracies):
                if acc*100 >= level:
                    granularity_indices.append(idx)
                    break
            else:
                granularity_indices.append(None)

        return granularity_indices

    def _personal_on_test(self):
        """
        Evaluates the personal models of all clients on the test data.
        """
        num_samples: list[int] = []
        tot_correct: list[float] = []
        for client in self.clients:
            result = client.eval_on_test()
            tot_correct.append(result['acc']*1.0)
            num_samples.append(result['num'])

        test_acc = sum(tot_correct)*1.0 / sum(num_samples)
        accs = [a / n for a, n in zip(tot_correct, num_samples)]
        self.metrics["test_personal_accs"].append(test_acc)
        self.logger.info(f'Std  Personal Test   ACC: {np.std(accs)}')
        self.logger.info(f'Mean Personal Test   ACC: {test_acc}')

    def _personal_on_train(self):
        """
        Evaluates the personal models of all clients on the training data.
        """
        num_samples: list[int] = []
        losses: list[float] = []
        for c in self.clients:
            result = c.eval_on_train()
            num_samples.append(result['num'])
            losses.append(result['losses']*1.0)
        train_loss = sum(losses)*1.0 / sum(num_samples)
        self.metrics["losses"].append(train_loss)
        self.logger.info(f'Mean Personal Train Loss: {train_loss}')

    def personal_evaluation(self):
        """
        Evaluates the personal models of all clients on both test and train data.
        """
        self._personal_on_train()
        self._personal_on_test()
    
    def server_evaluation(self):
        """
        Evaluates the global model on the test data of all clients.
        """
        tot_correct: list[float] = []
        num_samples: list[int] = []

        for client in self.clients:
            result = super().evaluation_testset(
                model=self.global_model,
                dataloader=client.load_data(flag='test'),
                device=self.device,
            )
            tot_correct.append(result['acc']*1.0)
            num_samples.append(result['num'])

        test_acc = sum(tot_correct)*1.0 / sum(num_samples)
        accs = [a / n for a, n in zip(tot_correct, num_samples)]
        self.metrics["test_traditional_accs"].append(test_acc)
        self.logger.info(f'Mean Global   Test   ACC: {test_acc}')
        self.logger.info(f'Std  Global   Test   ACC: {np.std(accs)}')

    def evaluate(self):
        """
        Evaluates the global model on the test data.

        configs:
            i: The current round number.
        """
        if self.current_iter%self.eval_gap == 0:
            self.logger.info('')
            self.logger.info(f'-------------Round number: {str(self.current_iter).zfill(4)}-------------')
            self.personal_evaluation()
            self.server_evaluation()

    def early_stopping(self, acc_lss: list[list[float]], div_value: float = None) -> bool:
        """
        Checks for convergence based on accuracy and standard deviation.

        configs:
            acc_lss: A list of accuracy lists for different metrics.
            div_value: The maximum allowed standard deviation of recent accuracies.

        Returns:
            True if convergence is detected, False otherwise.
        """
        if self.patience is None: return False

        for acc_ls in acc_lss:
            if len(acc_ls) < 2: return False

            # Determine the index of the highest accuracy
            top_index = torch.topk(torch.tensor(acc_ls), 1).indices[0].item()
            # Get the most recent accuracy values, up to the number of patience steps
            recent_accuracies = acc_ls[-self.patience:]

            # Check if the highest accuracy was achieved more than 'patience' epochs ago
            self.logger.info(f'                Patience: {(len(acc_ls) - top_index)}')
            patience_condition = (len(acc_ls) - top_index) > self.patience
            # Check if the standard deviation of the recent accuracies is below 'div_value'
            std_condition = np.std(recent_accuracies) < div_value if div_value is not None else True
            
            # Convergence requires both conditions to be met, if both are specified
            if patience_condition and std_condition:
                self.logger.info('Early stopping condition met.')
                return True

        return False
    
    def train_clients(self):
        """
        Trains the models of selected clients.
        """

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

    def _train_batch(self, clients):
        """Trains a batch of clients sequentially within a thread."""
        for client in clients:
            client.train()

    def models_sumary(self):
        """
        Sumary the models of server.
        """
        testloader = self.clients[0].load_data(flag='test')
        for model, name in zip([self.global_model], ['server_model']):
            ModelSummary(
                model=model, 
                save_path=os.path.join(self.model_info_path, f'{name}.svg'),
                dataloader=testloader
            )()
    
    def train(self):
        """
        Executes the federated learning training process.
        """
        self.make_logger(name='  SERVER  ', path=self.log_path)
        self.get_model()
        self.set_clients()
        self.models_sumary()
        for i in range(self.iterations+1):
            self.current_iter = i
            s_t = time.time()
            self.select_clients()
            self.send_models()
            self.evaluate()
            self.train_clients()
            self.receive_models()
            self.server_aggregation()
            self.metrics["time_per_iter"].append(time.time() - s_t)
            self.logger.info(f'               Time cost: {self.metrics["time_per_iter"][-1]}')
            self.save_models()
            if self.early_stopping(acc_lss=[self.metrics["test_personal_accs"]]): 
                break

        self.logger.info('')
        self.logger.info('-'*50)
        self.logger.info(f'Best accuracy: {max(self.metrics["test_personal_accs"])}')
        self.logger.info(f'Average time cost per round: {sum(self.metrics["time_per_iter"][1:])/len(self.metrics["time_per_iter"][1:])}')
        
        self.save_results()

class BaseClient(SharedMethods):
    """
    Base client class for federated learning.
    """
    def __init__(self, configs: dict, id: int, model: nn.Module, times: int):
        """
        Initializes the client.

        configs:
            configs: A dictionary of arguments.
            id: The client ID.
        """
        self.set_configs(configs=configs, id=id, times=times) 
        self.mkdir()
        self.model = copy.deepcopy(model)

        self.metrics = {
            'train_time': [],
            'send_time': [],
            'accs': [],
            'losses': [],
        }

        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer()

        self.train_file = os.path.join(self.dataset_path, 'train/', str(self.id) + '.npz')
        self.test_file = os.path.join(self.dataset_path, 'test/', str(self.id) + '.npz')
        trainset = self.load_data(flag='train')
        testset = self.load_data(flag='test')
        self.train_samples = len(trainset)
        self.test_samples = len(testset)

        self.make_logger(name=f'CLIENT_{str(self.id).zfill(3)}', path=self.log_path)
        self.save = False
    
    def get_loss(self):
        """
        Returns the loss function.
        """
        return getattr(__import__('losses'), self.loss)()

    def get_optimizer(self):
        """
        Returns the optimizer.
        """
        return getattr(__import__('optimizers'), self.optimizer)(self.model.parameters(), lr=self.learning_rate)

    def load_data(self, flag: str = 'train') -> DataLoader:
        """
        Loads the training or test data for the client.

        configs:
            flag: 'train' or 'test'.

        Returns:
            A DataLoader object for the specified data.
        """
        file = self.train_file if flag == 'train' else self.test_file
        with open(file, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()

        x = torch.Tensor(data['x']).type(torch.float32)
        y = torch.Tensor(data['y']).type(torch.int64)

        data = [(x, y) for x, y in zip(x, y)]
        return DataLoader(data, self.batch_size, drop_last=True, shuffle=True)
        
    def initialize_local(self, model: nn.Module):
        """
        Initializes the local model with the global model parameters.

        configs:
            model: The global model.
        """
        # NOTE: deepcopy will not work since the model parameters are binded to the optimizer
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def eval_on_test(self) -> tuple[float, int, float]:
        """
        Evaluates the local model on the test data.

        Returns:
            A tuple containing the accuracy, number of samples, and AUC.
        """
        result = super().evaluation_testset(
            model=self.model,
            dataloader=self.load_data(flag='test'),
            device=self.device,
        )

        acc = result['acc']/result['num']
        self.logger.info(f'Test Accuracy: {acc*100:.4f}%')
        self.metrics['accs'].append(acc)
        
        return result

    def eval_on_train(self) -> tuple[float, int]:
        """
        Evaluates the local model on the training data.

        Returns:
            A tuple containing the average loss and number of samples.
        """
        result = super().evaluation_trainset(
            model=self.model, 
            dataloader=self.load_data(flag='train'), 
            loss=self.loss, 
            device=self.device
        )
        loss = result['losses'] / result['num']
        self.logger.info(f'Train Loss: {loss:.4f}')
        self.metrics['losses'].append(loss)
        return result

    def train(self):
        """
        Trains the local model on the training data.
        """
        trainloader = self.load_data(flag='train')
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
                loss = self._train_loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self._optim_step()

        self.metrics['train_time'].append(time.time() - start_time)
    
    def _train_loss(self, output, y):
        """
        Calculates the loss of the model.

        Args:
            output: The model's output.
            y: The target labels.

        Returns:
            The loss value.
        """
        return self.loss(output, y)

    def _optim_step(self):
        """
        Performs an optimization step.
        """
        self.optimizer.step()        
    
    def save_model(self):
        """
        Saves the local model.
        """
        if self.save:
            path = os.path.join(self.model_path, f'client_{str(self.id).zfill(3)}.pt')
            torch.save(self.model, path)
            self.logger.info(f'Model saved to {path}')
    
    def save_results(self):
        """
        Saves the accuracy and loss results as a single Polars DataFrame.
        """
        pl_df = pl.DataFrame(self.metrics)
        path = os.path.join(self.result_path, f'client_{str(self.id).zfill(3)}.csv')
        pl_df.write_csv(path)
        self.logger.info(f'Results saved to {path}')