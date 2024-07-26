# Federated Learning Library

## Installation (Linux)
```bash
pip install --upgrade pip
```
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
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage
```bash
python main.py
```

## Implemented Frameworks
| Name | Paper | URL |
|------|-------|-----|
| LocalOnly | | |
| FedAvg | Communication-Efficient Learning of Deep Networks from Decentralized Data | [Arxiv](https://arxiv.org/abs/1602.05629) |
| FedProx | Federated Optimization in Heterogeneous Networks | [Arxiv](https://arxiv.org/abs/1812.06127) |
| FedCAC | Critical Area Collaboration in Federated Learning | [OpenAccess](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Bold_but_Cautious_Unlocking_the_Potential_of_Personalized_Federated_Learning_ICCV_2023_paper.html) |


## Acknowledgements
This codebase was adapted from [PFLlib](https://github.com/TsingZ0/PFLlib).