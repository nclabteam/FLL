# Federated Learning Library

## Installation (Linux)
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
For more hyperameters, see `./utils/options.py`.  
For more frameworks, see `./frameworks/__init__.py`

## Inspiration
This project is inspired by [PFLlib](https://github.com/TsingZ0/PFLlib).