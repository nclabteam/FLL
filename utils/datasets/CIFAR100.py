import torchvision
from .base import DatasetGenerator
import torchvision.transforms as transforms

class CIFAR100_Generator(DatasetGenerator):
    """
    A generator class for the CIFAR100 dataset.

    This class inherits from DatasetGenerator and provides specific logic for
    downloading and processing CIFAR100 data.
    """
    def __init__(self, *args, **kwargs):
        defaults = {
            'class_per_client': 10,
            'plot_ylabel_step': 20
        }

        for key, default_value in defaults.items():
            if key not in kwargs or kwargs[key] is None:
                kwargs[key] = default_value
                
        super().__init__(*args, **kwargs)

    def download(self) -> tuple[torchvision.datasets.CIFAR100, torchvision.datasets.CIFAR100]:
        """
        Downloads the CIFAR100 dataset.

        Returns:
            tuple[torchvision.datasets.CIFAR100, torchvision.datasets.CIFAR100]:
                A tuple containing the train and test CIFAR100 datasets.
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(), 
                transforms.Normalize(
                    (0.5, 0.5, 0.5), 
                    (0.5, 0.5, 0.5)
                )
            ]
        )

        trainset = torchvision.datasets.CIFAR100(
            root=self.rawdata_path, 
            train=True, 
            download=True, 
            transform=transform
        )
        testset = torchvision.datasets.CIFAR100(
            root=self.rawdata_path, 
            train=False, 
            download=True, 
            transform=transform
        )
        
        self.batch_size_train_cal = len(trainset.data)
        self.batch_size_test_cal = len(testset.data)
        return trainset, testset