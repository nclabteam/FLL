# Federated Learning Library

## Installation (Linux)
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

## Usage
```bash
python main.py
```

## Implemented Frameworks
| Name | Venue | Year | Paper | URL |
|------|-------|------| ----- | ---|
| LocalOnly | | | | | 
| Central | | | | |
| FedAvg | AISTATS | 2017 | Communication-Efficient Learning of Deep Networks from Decentralized Data | [Arxiv](https://arxiv.org/abs/1602.05629) |
| FedAtt | IJCNN | 2019 | Learning Private Neural Language Modeling with Attentive Aggregation | [Arxiv](https://arxiv.org/abs/1812.07108) |
| FedProx | MLsys | 2020| Federated Optimization in Heterogeneous Networks | [Arxiv](https://arxiv.org/abs/1812.06127) |
| FedALA | AAAI | 2023 | FedALA: Adaptive Local Aggregation for Personalized Federated Learning | [Arxiv](https://arxiv.org/abs/2212.01197) |
| FedCAC | ICCV | 2023 | Critical Area Collaboration in Federated Learning | [Arxiv](https://arxiv.org/abs/2309.11103)[OpenAccess](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Bold_but_Cautious_Unlocking_the_Potential_of_Personalized_Federated_Learning_ICCV_2023_paper.html) |
| FedPolyak | CKAIA | 2024 | FedPolyak: Personalized Federated Learning using Poyak-Ruppert Averaging | |


## Acknowledgements
This codebase was adapted from [PFLlib](https://github.com/TsingZ0/PFLlib).