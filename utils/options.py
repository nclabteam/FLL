import os
import json
import torch
import argparse
from .general import increment_path

from rich.table import Table
from rich import box
from rich.terminal_theme import MONOKAI
from rich.console import Console

OPTIMIZERS = ['Adam', 'SGD', 'SPS', 'SLS']
DATASETS = ['cifar10', 'cifar100', 'tinyimagenet']
DATA_PARTITIONS = ['dir', 'pat', 'exdir']
# FRAMEWORKS = ['FedAvg', 'FedProx', 'LocalOnly', 'FedFlex', 'FedPolyak', 'FjORD', 'FedALA', 'FedMixout', 'FedEHFT', 'FedX2TPolyak', 'FedCAC', 'FedMX2TPolyak']
FRAMEWORK_PATH = os.path.abspath(os.path.join('frameworks', 'centralized'))
FRAMEWORKS = [os.path.splitext(file)[0] for file in os.listdir(FRAMEWORK_PATH) if os.path.isfile(os.path.join(FRAMEWORK_PATH, file))]
LOSSES = ['CEL', 'CrossEntropyLoss']
TOPOLOGIES = ['Ring', 'FullyConnected']
TOLERANCE = 1e-6
MODELS = [
    'FedAvgCNN', 
    'ResNet10', 'ResNet10Half', 'ResNet10Double', 
    'ResNet18', 'ResNet18Half', 'ResNet18Double', 
    'ResNet22', 
    'ResNet34', 'ResNet34Half', 'ResNet34Double'
]

none_or_str = lambda x: None if x == 'None' else x

class Options:
    def __init__(self, ROOT):
        self.ROOT = ROOT
    
    def parse_options(self):
        parser = argparse.ArgumentParser()

        # general 
        parser.add_argument('--times', type=int, default=1, help='number of times to run the experiment')
        parser.add_argument('--seed', type=int, default=941)
        parser.add_argument('--device', type=str, default="cuda", choices=["cpu", "cuda"])
        parser.add_argument('--device_id', type=str, default="0")
        parser.add_argument('--workers', type=int, default=1, help='number of workers for dataloader')

        # save path
        parser.add_argument('--project', type=str, default=os.path.join(self.ROOT, 'runs'), help='project name')
        parser.add_argument('--name', type=str, default='exp', help='name of this experiment')
        parser.add_argument('--sep', type=str, default='', help='separator for name')
            
        # dataset
        parser.add_argument('--dataset', type=str, default='cifar100', help='name of dataset', choices=DATASETS)
        parser.add_argument('--num_clients', type=int, default=20, help='number of clients')
        parser.add_argument('--train_ratio', type=float, default=0.75, help='train ratio')
        parser.add_argument('--batch_size', type=int, default=10, help='batch size')
        parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
        parser.add_argument('--iid', action='store_true', default=False, help='niid')
        parser.add_argument('--balance', action='store_true', default=False, help='balance')
        parser.add_argument('--partition', type=str, default='dir', help='data partition', choices=DATA_PARTITIONS)
        parser.add_argument('--class_per_client', type=int, default=None, help='number of classes per client')
        parser.add_argument('--plot_ylabel_step', type=int, default=None, help='plot ylabel step')

        # server
        parser.add_argument('--model', type=str, default='FedAvgCNN', help='model name', choices=MODELS)
        parser.add_argument('--framework', type=str, default='FedAvg', choices=FRAMEWORKS)
        parser.add_argument('--iterations', type=int, default=2000)
        parser.add_argument('--join_ratio', type=float, default=1.0, help="Ratio of clients per round")
        parser.add_argument('--random_join_ratio', action='store_true', default=False, help="Random ratio of clients per round")
        parser.add_argument('--patience', type=int, default=None, help="Patience for early stopping")
        parser.add_argument('--eval_gap', type=int, default=1, help="Rounds gap for evaluation")
        
        # client
        parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer name', choices=OPTIMIZERS)
        parser.add_argument('--learning_rate', type=float, default=0.005, help="Local learning rate")
        parser.add_argument('--epochs', type=int, default=1, help="Multiple update steps in one local epoch.")
        parser.add_argument('--loss', type=str, default='CrossEntropyLoss', help='loss function', choices=LOSSES)

        # decentralized
        parser.add_argument('--topology', type=str, default=None, help='topology', choices=TOPOLOGIES)
        
        # FedProx
        parser.add_argument('--mu', type=float, default=None, help='proximal term')

        # FedCAC
        parser.add_argument('--tau', type=float, default=None)

        # FedCAC
        parser.add_argument('--beta', type=float, default=None)
        
        self.args = parser.parse_args()
        return self
    
    def add_args(self, name, value):
        self.args.__dict__[name] = value
        
    def update_args(self, params: dict):
        """
        Update self.args with a given dictionary of parameters.

        Args:
            params (dict): Dictionary containing parameter updates.
        """
        for key, value in params.items():
            self.add_args(key, value)

    def update_if_none(self, params: dict):
        """
        Update self.args with a given dictionary of parameters only if they are None.

        Args:
            params (dict): Dictionary containing parameter updates.
        """
        for key, value in params.items():
            if getattr(self.args, key) is None:
                self.add_args(key, value)

    def _fix_framework_specific_param(self):
        if self.args.framework == "FedProx":
            self.update_if_none({'mu': 0.001})
        
        elif self.args.framework == 'FedCAC':
            self.update_if_none({'tau': 0.5, 'beta': 160})

    def _fix_save_path(self):
        path = increment_path(
            os.path.join(
                self.args.project, 
                self.args.name
            ), 
            exist_ok=False, 
            sep=self.args.sep
        )
        self.add_args('save_path', path)
        if not os.path.exists(path):
            os.makedirs(path)

    def _fix_device(self):
        if self.args.device == 'cuda' and not torch.cuda.is_available():
            print("cuda is not available. Using cpu instead.")
            self.add_args('device', 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.device_id

    def _fix_model(self):
        if self.args.model in ["FedAvgCNN", 'FjORDCNN']:
            if "cifar10" in self.args.dataset.lower():
                dim = 1600
                in_features = 3
            else:
                dim = 10816
                in_features = 3
            self.add_args('dim', dim)
            self.add_args('in_features', in_features)

    def _fix_optimizer(self):
        if self.args.framework == 'FedProx':
            self.add_args('optimizer', 'PerturbedGradientDescent')

    def fix_args(self):
        self._fix_save_path()
        self._fix_device()
        self._fix_optimizer()
        self._fix_framework_specific_param()
        self._fix_model()

        return self

    def save(self):
        """
        Saves all values in self.args to a file.
        """
        with open(os.path.join(self.args.save_path, 'config.json'), 'w') as f:
            json.dump(vars(self.args), f, indent=4)
    
    def display(self):
        """
        Displays all values in self.args using rich package.
        """

        table = Table(title="Experiment Arguments", box=box.ROUNDED)
        table.add_column("Argument", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        for arg in vars(self.args):
            table.add_row(arg, str(getattr(self.args, arg)))

        console = Console(record=True)
        console.print(table)
        console.save_svg(os.path.join(self.args.save_path, 'configs.svg'), theme=MONOKAI)