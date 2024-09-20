# Federated Learning Library

## Installation

### Linux
```bash
pip install virtualenv
```
```bash
virtualenv venv --python=python3.10
```
```bash
source venv/bin/activate
```
```bash
pip install --upgrade pip
```
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Window
```bash
pip install virtualenv
```
```bash
virtualenv venv --python=python3.10
```
```bash
.\venv\Scripts\activate
```
```bash
pip install --upgrade pip
```
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Usage
```bash
python main.py
```

---

## Customization

---

## Reproduce

### Benchmark

<details>
<summary>Configs</summary>
<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Param</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=5>Dataset</td>
            <td>batch_size</td>
            <td>10</td>
        </tr>
        <tr>
            <td>alpha</td>
            <td>0.1</td>
        </tr>
        <tr>
            <td>iid</td>
            <td>False</td>
        </tr>
        <tr>
            <td>balance</td>
            <td>False</td>
        </tr>
        <tr>
            <td>partition</td>
            <td>dir</td>
        </tr>
        <tr>
            <td rowspan=2>Evaluation</td>
            <td>times</td>
            <td>5</td>
        </tr>
        <tr>
            <td>eval_gap</td>
            <td>1</td>
        </tr>
        <tr>
            <td rowspan=2>Model</td>
            <td>model</td>
            <td>FedAvgCNN</td>
        </tr>
        <tr>
            <td>dim</td>
            <td>1600</td>
        </tr>
        <tr>
            <td rowspan=3>Server</td>
            <td>num_clients</td>
            <td>100</td>
        </tr>
        <tr>
            <td>join_ratio</td>
            <td>0.1</td>
        </tr>
        <tr>
            <td>iterations</td>
            <td>2000</td>
        </tr>
        <tr>
            <td rowspan=5>Client</td>
            <td>optimizer</td>
            <td>SGD</td>
        </tr>
        <tr>
            <td>momentum</td>
            <td>0.9</td>
        </tr>
        <tr>
            <td>learning_rate</td>
            <td>0.005</td>
        </tr>
        <tr>
            <td>epochs</td>
            <td>1</td>
        </tr>
        <tr>
            <td>loss</td>
            <td>CEL</td>
        </tr>
        <tr>
            <td rowspan=5>FedALA</td>
            <td>eta</td>
            <td>1.0</td>
        </tr>
        <tr>
            <td>data_rand_percent</td>
            <td>0.8</td>
        </tr>
        <tr>
            <td>p</td>
            <td>2</td>
        </tr>
        <tr>
            <td>threshold</td>
            <td>1.1</td>
        </tr>
        <tr>
            <td>local_patience</td>
            <td>10</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary>Results</summary>

|            | **Cifar100**        |                    |**TinyImagenet**       |                     |
|------------|---------------------|--------------------|-----------------------|---------------------|
|            | **Personalization** | **Generalization** | **Personalization**   | **Generalization**  |
| **FedAvg** | 32.66±0.528         | 32.4±0.354         | ±                     | ±                   |
| **FedALA** | 35.69±0.3           | 30.964±0.45        | ±                     | ±                   |

</details>

---

### FedALA

#### Table 2

<details>
<summary>Configs</summary>
<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Param</th>
            <th>Value</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=5>Dataset</td>
            <td>batch_size</td>
            <td>10</td>
        </tr>
        <tr>
            <td>alpha</td>
            <td>0.1</td>
        </tr>
        <tr>
            <td>iid</td>
            <td>False</td>
        </tr>
        <tr>
            <td>balance</td>
            <td>False</td>
        </tr>
        <tr>
            <td>partition</td>
            <td>dir</td>
        </tr>
        <tr>
            <td rowspan=2>Evaluation</td>
            <td>times</td>
            <td>5</td>
        </tr>
        <tr>
            <td>eval_gap</td>
            <td>1</td>
        </tr>
        <tr>
            <td rowspan=2>Model</td>
            <td>model</td>
            <td>FedAvgCNN</td>
        </tr>
        <tr>
            <td>dim</td>
            <td>1600</td>
        </tr>
        <tr>
            <td rowspan=3>Server</td>
            <td>num_clients</td>
            <td>20</td>
        </tr>
        <tr>
            <td>join_ratio</td>
            <td>1</td>
        </tr>
        <tr>
            <td>iterations</td>
            <td>2000</td>
        </tr>
        <tr>
            <td rowspan=5>Client</td>
            <td>optimizer</td>
            <td>SGD</td>
        </tr>
        <tr>
            <td>momentum</td>
            <td>0.0</td>
        </tr>
        <tr>
            <td>learning_rate</td>
            <td>0.005</td>
        </tr>
        <tr>
            <td>epochs</td>
            <td>1</td>
        </tr>
        <tr>
            <td>loss</td>
            <td>CEL</td>
        </tr>
        <tr>
            <td rowspan=5>FedALA</td>
            <td>eta</td>
            <td>1.0</td>
        </tr>
        <tr>
            <td>data_rand_percent</td>
            <td>0.8</td>
        </tr>
        <tr>
            <td>p</td>
            <td>2</td>
        </tr>
        <tr>
            <td>threshold</td>
            <td>1.1</td>
        </tr>
        <tr>
            <td>local_patience</td>
            <td>10</td>
        </tr>
    </tbody>
</table>
</details>

<details>
<summary>Results</summary>

|                        | **Cifar10** | **Cifar100** | **TINY**   | **TINY***  | **AG News** |
|------------------------|-------------|--------------|------------|------------|-------------|
| **FedAvg (Paper)**     | 59.16±0.47  | 31.89±0.47   | 19.46±0.20 | 19.45±0.13 | 79.57±0.17  |
| **FedProx (Paper)**    | 59.21±0.40  | 31.99±0.41   | 19.37±0.22 | 19.27±0.23 | 79.35±0.23  |
| **FedAvg-C (Paper)**   | 90.34±0.01  | 51.80±0.02   | 30.67±0.08 | 36.94±0.10 | 95.89±0.25  |
| **FedProx-C (Paper)**  | 90.33±0.01  | 51.84±0.07   | 30.77±0.13 | 38.78±0.52 | 96.10±0.25  |
| **Per-FedAvg (Paper)** | 87.74±0.19  | 44.28±0.33   | 25.07±0.07 | 21.81±0.54 | 93.27±0.25  |
| **FedRep (Paper)**     | 90.40±0.24  | 52.39±0.35   | 37.27±0.20 | 39.95±0.61 | 96.28±0.18  |
| **pFedMe (Paper)**     | 88.09±0.32  | 47.34±0.46   | 26.93±0.19 | 33.44±0.33 | 91.41±0.17  |
| **Ditto (Paper)**      | 90.59±0.01  | 52.87±0.64   | 32.15±0.04 | 35.92±0.43 | 95.80±0.12  |
| **FedAMP (Paper)**     | 88.70±0.18  | 47.69±0.49   | 27.99±0.11 | 29.11±0.85 | 94.18±0.09  |
| **FedPHP (Paper)**     | 88.92±0.02  | 50.52±0.16   | 35.69±3.26 | 29.90±0.51 | 94.38±0.12  |
| **FedFomo (Paper)**    | 88.06±0.02  | 45.39±0.45   | 26.33±0.22 | 26.84±0.11 | 95.84±0.12  |
| **APPLE (Paper)**      | 89.37±0.11  | 53.22±0.20   | 35.04±0.42 | 34.01±0.42 | 95.86±0.17  |
| **PartialFed (Paper)** | 87.38±0.08  | 48.16±0.20   | 32.58±0.16 | 37.50±0.16 | 95.80±0.16  |
| **FedALA (Paper)**     | 90.67±0.03  | 55.92±0.03   | 40.54±0.02 | 41.94±0.05 | 96.52±0.08  |
| **FedALA**             |             | 56.32±0.32   |            |            |             |

</details>

---
## Acknowledgements
This codebase was adapted from [PFLlib](https://github.com/TsingZ0/PFLlib).