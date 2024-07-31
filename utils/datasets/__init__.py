from .base import DatasetGenerator
from .CIFAR10 import CIFAR10_Generator
from .CIFAR100 import CIFAR100_Generator
from .TinyImagenet import TINYIMAGENET_Generator

DATASETS = ['cifar10', 'cifar100', 'tinyimagenet']
DATA_PARTITIONS = ['dir', 'pat', 'exdir']

class DatasetFactory:
    def __init__(self, args):
        self.dataset = args.dataset
        self.num_clients = args.num_clients
        self.batch_size = args.batch_size
        self.train_ratio = args.train_ratio
        self.alpha = args.alpha
        self.niid = not args.iid
        self.balance = args.balance
        self.partition = args.partition
        self.class_per_client = args.class_per_client
        self.plot_ylabel_step = args.plot_ylabel_step
        self.strat = args.strat
    
    def __call__(self):
        generator = globals()[f"{self.dataset.upper()}_Generator"](
            dataset_name=self.dataset,
            num_clients=self.num_clients,
            batch_size=self.batch_size,
            train_ratio=self.train_ratio,
            alpha=self.alpha,
            niid=self.niid,
            balance=self.balance,
            partition=self.partition,
            class_per_client=self.class_per_client,
            plot_ylabel_step=self.plot_ylabel_step,
            strat=self.strat,
        )
        generator.generate_data()
        self.num_classes = generator.num_classes
        self.path = generator.dir_path
        return self
