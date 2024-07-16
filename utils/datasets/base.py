import os
import yaml
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List, Tuple

class DatasetGenerator:
    """
    A class to generate partitioned datasets for federated learning.

    This class provides functionality to download, partition, and save datasets
    for different federated learning scenarios. It supports various partitioning
    strategies, including:

    - **PAT (Pathologycal Heterogeneous):** 
      Each client receives a fixed number of classes, and data is distributed
      proportionally within those classes.
    - **DIR (Dirichlet Distribution):** Each client receives data from all classes
      with proportions determined by a Dirichlet distribution.
    - **EXDIR (Extended Dirichlet):** This method allows for controlling the
      number of classes per client by assigning classes randomly and then
      applying the Dirichlet distribution for data allocation within those
      assigned classes.

    Attributes:
        dataset_name (str): The name of the dataset.
        num_clients (int): The number of clients to partition the dataset for.
        batch_size (int): The batch size for data loading.
        train_ratio (float): The ratio of data to be used for training.
        alpha (float): The concentration parameter for the Dirichlet distribution.
        niid (bool): Whether to create non-IID partitions.
        balance (bool): Whether to balance the number of samples per class.
        partition (str): The partitioning strategy to use ('pat', 'dir', 'exdir').
        class_per_client (int): The number of classes to assign to each client (only for 'exdir').
        plot_ylabel_step (int): The step size for y-axis labels in the distribution plots.
        dir_path (str): The directory path for storing the partitioned data.
        config_path (str): The path to the configuration file.
        train_path (str): The path to the training data directory.
        test_path (str): The path to the test data directory.
        rawdata_path (str): The path to the raw data directory.
        num_samples (dict): Stores the number of train and test samples for each client.

    Methods:
        check(): Checks if the dataset has already been generated.
        partition_data_pat(): Partitions data using the PAT strategy.
        partition_data_dir(): Partitions data using the DIR strategy.
        partition_data_exdir(): Partitions data using the EXDIR strategy.
        split_data(): Splits the partitioned data into train and test sets.
        save_file(): Saves the partitioned data and configuration.
        download(): Downloads the raw dataset (to be implemented by subclasses).
        load_and_process_data(): Loads and processes the raw dataset.
        normalize_sizes(): Normalizes the sample sizes to a specified range.
        _plot_helper(): Helper function to plot data distribution.
        plot(): Plots the data distribution.
        generate_data(): Generates the partitioned dataset.
    """
    def __init__(
            self, 
            num_clients: int, 
            dataset_name: str = None,
            batch_size: int = 10, 
            train_ratio: float = 0.75, 
            alpha: float = 0.1, 
            niid: bool = True, 
            balance: bool = False, 
            partition: str = 'pat', 
            class_per_client: int = None,
            plot_ylabel_step: int = None
        ):
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.alpha = alpha
        self.niid = niid
        self.balance = balance
        self.partition = partition
        self.class_per_client = class_per_client
        self.plot_ylabel_step = plot_ylabel_step

        self.dir_path = os.path.join(
            'datasets',
            f'{dataset_name.upper()}',
            f'partition_{partition}-clients_{num_clients}-niid_{str(niid)}-balance_{str(balance)}-batch_{batch_size}-classpc_{class_per_client}-alpha_{str(alpha).replace(".", "")}-train_{str(train_ratio).replace(".", "")}')
        self.config_path = os.path.join(self.dir_path, "config.yaml")
        self.train_path = os.path.join(self.dir_path, "train/")
        self.test_path = os.path.join(self.dir_path, "test/")
        self.rawdata_path = os.path.join('datasets', f'{dataset_name.upper()}', "rawdata/")
        
        random.seed(1)
        np.random.seed(1)
        
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

    def check(self) -> bool:
        """
        Checks if the dataset has already been generated.

        Returns:
            bool: True if the dataset exists, False otherwise.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            if config['num_clients'] == self.num_clients and \
                config['non_iid'] == self.niid and \
                config['balance'] == self.balance and \
                config['partition'] == self.partition and \
                config['alpha'] == self.alpha and \
                config['batch_size'] == self.batch_size:
                # print(f"Dataset already generated: {self.dir_path}")
                self.num_classes = config['num_classes']
                return True

        if not os.path.exists(self.train_path):
            os.makedirs(self.train_path)
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path)

        return False

    def partition_data_pat(self, dataset_content: np.ndarray, dataset_label: np.ndarray, num_classes: int) -> tuple[list[np.ndarray], list[np.ndarray], list[list[tuple[int, int]]]]:
        X = [[] for _ in range(self.num_clients)]
        y = [[] for _ in range(self.num_clients)]
        statistic = [[] for _ in range(self.num_clients)]
        
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = [idxs[dataset_label == i] for i in range(num_classes)]
        dataidx_map = {}

        class_num_per_client = [self.class_per_client for _ in range(self.num_clients)]
        for i in range(num_classes):
            selected_clients = [client for client in range(self.num_clients) if class_num_per_client[client] > 0]
            selected_clients = selected_clients[:int(np.ceil((self.num_clients / num_classes) * self.class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            num_samples = [int(num_per) for _ in range(num_selected_clients - 1)] if self.balance else \
                np.random.randint(max(num_per / 10, len(dataset_label) / self.num_clients / 2 / num_classes), num_per, num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

        for client in range(self.num_clients):
            idxs = dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]
            for i in np.unique(y[client]):
                statistic[client].append((int(i), int(sum(y[client] == i))))

        return X, y, statistic

    def partition_data_dir(self, dataset_content: np.ndarray, dataset_label: np.ndarray, num_classes: int) -> tuple[list[np.ndarray], list[np.ndarray], list[list[tuple[int, int]]]]:
        """
        Partitions data using the Dirichlet Distribution (DIR) strategy.

        Args:
            dataset_content (np.ndarray): The dataset content (e.g., images).
            dataset_label (np.ndarray): The dataset labels.
            num_classes (int): The number of classes in the dataset.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], list[list[tuple[int, int]]]]:
                A tuple containing:
                - X: A list of data arrays for each client.
                - y: A list of label arrays for each client.
                - statistic: A list of class-wise sample counts for each client.
        """
        X = [[] for _ in range(self.num_clients)]
        y = [[] for _ in range(self.num_clients)]
        statistic = [[] for _ in range(self.num_clients)]
        
        min_size = 0
        N = len(dataset_label)
        dataidx_map = {}

        while min_size < int(self.batch_size / (1 - self.train_ratio)):
            idx_batch = [[] for _ in range(self.num_clients)]
            for k in range(num_classes):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
                proportions = np.array([p * (len(idx_j) < N / self.num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(self.num_clients):
            dataidx_map[j] = idx_batch[j]

        for client in range(self.num_clients):
            idxs = dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]
            for i in np.unique(y[client]):
                statistic[client].append((int(i), int(sum(y[client] == i))))

        return X, y, statistic

    def partition_data_exdir(self, dataset_content: np.ndarray, dataset_label: np.ndarray, num_classes: int) -> tuple[list[np.ndarray], list[np.ndarray], list[list[tuple[int, int]]]]:
        """
        Partitions data using the Extended Dirichlet (EXDIR) strategy.

        Args:
            dataset_content (np.ndarray): The dataset content (e.g., images).
            dataset_label (np.ndarray): The dataset labels.
            num_classes (int): The number of classes in the dataset.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], list[list[tuple[int, int]]]]:
                A tuple containing:
                - X: A list of data arrays for each client.
                - y: A list of label arrays for each client.
                - statistic: A list of class-wise sample counts for each client.
        """
        X = [[] for _ in range(self.num_clients)]
        y = [[] for _ in range(self.num_clients)]
        statistic = [[] for _ in range(self.num_clients)]

        cnt = 0
        min_size_per_label = 0
        min_require_size_per_label = max(self.class_per_client * self.num_clients // num_classes // 2, 1)
        clientidx_map = {}

        while min_size_per_label < min_require_size_per_label:
            cnt += 1
            print(f'Iter1: {cnt:,} | {min_size_per_label:,}/{min_require_size_per_label:,}', end='\r')
            for k in range(num_classes):
                clientidx_map[k] = []
            for i in range(self.num_clients):
                labelidx = np.random.choice(range(num_classes), self.class_per_client, replace=False)
                for k in labelidx:
                    clientidx_map[k].append(i)
            min_size_per_label = min([len(clientidx_map[k]) for k in range(num_classes)])

        dataidx_map = {}
        min_size = 0
        min_require_size = 10
        K = num_classes
        N = len(dataset_label)
        cnt = 0

        while min_size < min_require_size:
            cnt += 1
            print(f'Iter2: {cnt:,} | {min_size_per_label:,}/{min_require_size_per_label:,}', end='\r')
            idx_batch = [[] for _ in range(self.num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
                proportions = np.array([p * (len(idx_j) < N / self.num_clients and j in clientidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], self.num_clients - 1):
                        proportions[w] = len(idx_k)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(self.num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]

        for client in range(self.num_clients):
            idxs = dataidx_map[client]
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]
            for i in np.unique(y[client]):
                statistic[client].append((int(i), int(sum(y[client] == i))))

        return X, y, statistic

    def split_data(self, X: list[np.ndarray], y: list[np.ndarray]) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
        """
        Splits the partitioned data into train and test sets.

        Args:
            X (list[np.ndarray]): A list of data arrays for each client.
            y (list[np.ndarray]): A list of label arrays for each client.

        Returns:
            tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
                A tuple containing:
                - train_data: A list of dictionaries with 'x' and 'y' for training data.
                - test_data: A list of dictionaries with 'x' and 'y' for test data.
        """
        train_data, test_data = [], []
        self.num_samples = {'train': [], 'test': []}

        for i in range(len(y)):
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=self.train_ratio, shuffle=True)

            train_data.append({'x': X_train, 'y': y_train})
            self.num_samples['train'].append(len(y_train))
            test_data.append({'x': X_test, 'y': y_test})
            self.num_samples['test'].append(len(y_test))

        return train_data, test_data

    def save_file(self, train_data: list[dict[str, np.ndarray]], test_data: list[dict[str, np.ndarray]], num_classes: int, statistic: list[list[tuple[int, int]]]):
        """
        Saves the partitioned data and configuration.

        Args:
            train_data (list[dict[str, np.ndarray]]): A list of dictionaries with 'x' and 'y' for training data.
            test_data (list[dict[str, np.ndarray]]): A list of dictionaries with 'x' and 'y' for test data.
            num_classes (int): The number of classes in the dataset.
            statistic (list[list[tuple[int, int]]]): A list of class-wise sample counts for each client.
        """
        # Convert the statistic list into a nested dict
        statistic_dict = {}
        for client_id, stats in enumerate(statistic):
            statistic_dict[client_id] = {}
            for clss, count in stats:
                statistic_dict[client_id][clss] = count
        
        for _, inner_dict in statistic_dict.items():
            for key in range(num_classes):
                if key not in inner_dict:
                    inner_dict[key] = 0

        statistic_dict = {
            outer_key: {
                k: inner_dict[k] for k in sorted(inner_dict)
            } 
            for outer_key, inner_dict in statistic_dict.items()
        }
        
        config = {
            'num_clients': self.num_clients,
            'num_classes': num_classes,
            'non_iid': self.niid,
            'balance': self.balance,
            'partition': self.partition,
            'alpha': self.alpha,
            'batch_size': self.batch_size,
            'labels_per_clients': statistic_dict,  # Use the dict here
        }

        print("Saving to disk.\n")

        for idx, train_dict in enumerate(train_data):
            with open(self.train_path + str(idx) + '.npz', 'wb') as f:
                np.savez_compressed(f, data=train_dict)
        for idx, test_dict in enumerate(test_data):
            with open(self.test_path + str(idx) + '.npz', 'wb') as f:
                np.savez_compressed(f, data=test_dict)
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)

        print("Finish generating dataset.\n")

    def download(self):
        """
        Downloads the raw dataset.

        This method should be overridden by subclasses to handle specific dataset downloads.

        Returns:
            tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
                A tuple containing the train and test datasets.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

    def load_and_process_data(self, trainset, testset) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Loads and processes the raw dataset.

        This method loads the downloaded data, combines train and test sets, and
        returns the processed data and the number of classes.

        Args:
            trainset: The training dataset.
            testset: The test dataset.

        Returns:
            tuple[np.ndarray, np.ndarray, int]:
                A tuple containing:
                - dataset_image: The combined dataset content (e.g., images).
                - dataset_label: The combined dataset labels.
                - num_classes: The number of classes in the dataset.
        """
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size_train_cal, shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size_test_cal, shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        num_classes = len(set(dataset_label))

        return dataset_image, dataset_label, num_classes

    def normalize_sizes(self, sizes: List[int], min_size: int, max_size: int) -> List[float]:
        """
        Normalizes the sample sizes to a specified range.

        Args:
            sizes (list): A list of sample sizes.
            min_size (int): The minimum desired size.
            max_size (int): The maximum desired size.

        Returns:
            list: Normalized sample sizes.
        """
        normalized = (sizes - min(sizes)) / max(sizes - min(sizes))
        return normalized * (max_size - min_size) + min_size

    def _plot_helper(self, data: List[List[Tuple[int, int]]], path: str, num_classes: int, normalize: bool = False, min_size: int = 30, max_size: int = 1500):
        """
        Helper function to plot data distribution with optional normalization.

        Args:
            data (list): A list of client data, where each element is a list of (label, sample_size) tuples.
            path (str): Path to save the plot.
            num_classes (int): Number of classes.
            normalize (bool, optional): Whether to normalize sample sizes. Defaults to False.
            min_size (int, optional): Minimum size for normalization. Defaults to 30.
            max_size (int): Maximum size for normalization. Defaults to 1500.
        """
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({'font.size': 20})  # Update font size
        plt.rcParams['axes.facecolor'] = 'white'

        # Flatten the list of samples to get the sizes for normalization
        if normalize:
            all_sizes = [sample for client in data for _, sample in client]
            normalized_sizes = self.normalize_sizes(
                sizes=np.array(all_sizes), 
                min_size=min_size, 
                max_size=max_size
            )

        # Loop through clients and their labels
        bubble_idx = 0
        for client_id, client_samples in enumerate(data):
            for label, sample_size in client_samples:
                if normalize:
                    size = normalized_sizes[bubble_idx]
                    bubble_idx += 1
                else:
                    size = sample_size
                plt.scatter(
                    client_id, 
                    label, 
                    s=size, 
                    c='red', 
                    alpha=0.5
                )  

        # Removing the outer box
        ax = plt.gca()  # Get current axes
        ax.spines['top'].set_visible(False)    # Hide the top spine
        ax.spines['right'].set_visible(False)  # Hide the right spine
        ax.spines['left'].set_visible(False)   # Hide the left spine
        ax.spines['bottom'].set_visible(False) # Hide the bottom spine

        plt.xlabel('Client IDs')
        plt.ylabel('Class IDs')
        plt.xticks(range(len(data)))  # Adjust x-ticks to match client count
        plt.yticks(range(num_classes // self.plot_ylabel_step))
        plt.grid(False)

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        # plt.show()

    def plot(self, statistic: list[list[tuple[int, int]]], num_classes):
        """
        Plots the data distribution.

        This method generates two plots:
        - 'distribution.png': Shows the data distribution with normalized sizes.
        - 'distribution_normalized.png': Shows the data distribution without normalization.

        Args:
            statistic (list[list[tuple[int, int]]]): A list of class-wise sample counts for each client.
            num_classes (int): The number of classes in the dataset.
        """
        self._plot_helper(
            data=statistic, 
            path=os.path.join(
                self.dir_path, 
                'distribution.png'
            ),
            num_classes=num_classes, 
            normalize=True, 
            min_size=30, 
            max_size=1500
        )
        self._plot_helper(
            data=statistic, 
            path=os.path.join(
                self.dir_path, 
                'distribution_normalized.png'
            ),
            num_classes=num_classes
        )

    def generate_data(self):
        """
        Generates the partitioned dataset.

        This method orchestrates the entire process of downloading, partitioning,
        and saving the dataset.
        """
        if self.check(): return
        trainset, testset = self.download()
        dataset_image, dataset_label, num_classes = self.load_and_process_data(trainset, testset)
        self.num_classes = num_classes

        if not self.niid:
            self.partition = 'pat'
            self.class_per_client = num_classes

        partition_method = getattr(self, f"partition_data_{self.partition}", None)
        if partition_method is None:
            raise ValueError("Unsupported partition method.")
        
        X, y, statistic = partition_method(dataset_content=dataset_image, dataset_label=dataset_label, num_classes=num_classes)
        self.plot(statistic=statistic, num_classes=num_classes)

        train_data, test_data = self.split_data(X, y)

        print(f'Number of classes: {num_classes}')
        print("Total number of samples:", sum(self.num_samples['train'] + self.num_samples['test']))
        print("The number of train samples:", self.num_samples['train'])
        print("The number of test samples:", self.num_samples['test'])
        print()

        self.save_file(train_data, test_data, num_classes, statistic)
