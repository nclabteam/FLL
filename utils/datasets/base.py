import os
import yaml
import numpy as np
import random
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import List, Tuple

class DatasetGenerator:
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
            plot_ylabel_step: int = None,
            strat: str = 'personalization' 
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
        self.strat = strat

        self.dir_path = os.path.join(
            'datasets',
            f'{dataset_name.upper()}',
            f'partition_{partition}-clients_{num_clients}-niid_{str(niid)}-balance_{str(balance)}-batch_{batch_size}-classpc_{class_per_client}-alpha_{str(alpha).replace(".", "")}-train_{str(train_ratio).replace(".", "")}-strat_{strat}')
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
                config['batch_size'] == self.batch_size and \
                config['strat'] == self.strat:
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

    def save_file(self, train_data: list[dict[str, np.ndarray]], test_data: list[dict[str, np.ndarray]], num_classes: int, statistic: list[list[tuple[int, int]]]):
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
            'strat': self.strat,
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
        raise NotImplementedError("This method should be overridden by subclasses")

    def load_and_process_data(self, trainset, testset) -> tuple[np.ndarray, np.ndarray, int]:
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset), shuffle=False)

        # Load and process trainset
        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data

        dataset_image = trainset.data.cpu().detach().numpy()
        dataset_label = trainset.targets.cpu().detach().numpy()

        # If testset is provided, load and process it
        if testset is not None:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=len(testset), shuffle=False)
            for _, test_data in enumerate(testloader, 0):
                testset.data, testset.targets = test_data

            dataset_image = np.concatenate(
                (dataset_image, testset.data.cpu().detach().numpy()))
            dataset_label = np.concatenate(
                (dataset_label, testset.targets.cpu().detach().numpy()))

        num_classes = len(set(dataset_label))

        return dataset_image, dataset_label, num_classes

    def normalize_sizes(self, sizes: List[int], min_size: int, max_size: int) -> List[float]:
        normalized = (sizes - min(sizes)) / max(sizes - min(sizes))
        return normalized * (max_size - min_size) + min_size

    def _plot_helper(self, data: List[List[Tuple[int, int]]], path: str, num_classes: int, normalize: bool = False, min_size: int = 30, max_size: int = 1500):
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
        plt.yticks(range(0, num_classes+1, self.plot_ylabel_step))
        plt.grid(False)

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        # plt.show()

    def plot(self, statistic: list[list[tuple[int, int]]], num_classes):
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

    def split_data(self, X: list[np.ndarray], y: list[np.ndarray]) -> tuple[list[dict[str, np.ndarray]], list[dict[str, np.ndarray]]]:
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

    def generate_data(self):
        if self.check(): return
        trainset, testset = self.download()
        partition_method = getattr(self, f"partition_data_{self.partition}", None)
        if partition_method is None:
            raise ValueError("Unsupported partition method.")
        if not self.niid: self.partition = 'pat'
        
        if self.strat == 'personalization':
            dataset_image, dataset_label, num_classes = self.load_and_process_data(trainset, testset)
            if not self.niid: self.class_per_client = num_classes
            X, y, statistic = partition_method(dataset_content=dataset_image, dataset_label=dataset_label, num_classes=num_classes)
            train_data, test_data = self.split_data(X, y)

        elif self.strat == 'generalization':
            train_dataset_image, train_dataset_label, num_classes = self.load_and_process_data(trainset, None) 
            if not self.niid: self.class_per_client = num_classes
            X, y, statistic = partition_method(dataset_content=train_dataset_image, dataset_label=train_dataset_label, num_classes=num_classes)
            train_data = [{'x': X[i], 'y': y[i]} for i in range(self.num_clients)]
            
            # Split testset into equal chunks (with adjusted batch size)
            test_data = []
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=min(len(testset), len(testset) // self.num_clients), shuffle=False)
            for i, test_batch in enumerate(testloader, 0):
                # Correct access to test_batch data
                data, targets = test_batch 
                test_data.append({'x': data.cpu().detach().numpy(), 'y': targets.cpu().detach().numpy()})

            self.num_samples = {
                'train': [len(y[i]) for i in range(self.num_clients)],
                'test': [len(test_data[i]['y']) for i in range(self.num_clients)] 
            }

        else:
            raise ValueError("Unsupported strategy: {}".format(self.strat))
        
        self.plot(statistic=statistic, num_classes=num_classes)
        self.num_classes = num_classes
        print(f'Number of classes: {num_classes}')
        print("Total number of samples:", sum(self.num_samples['train'] + self.num_samples['test']))
        print("The number of train samples:", self.num_samples['train'])
        print("The number of test samples:", self.num_samples['test'])
        print()
        self.save_file(train_data, test_data, num_classes, statistic)