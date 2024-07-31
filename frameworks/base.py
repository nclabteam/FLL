import os
import copy
import logging
from argparse import Namespace

class SharedMethods:
    def get_params(self, model):
        return sum(p.numel() for p in model.parameters())
    
    def make_logger(self, name, path):
        """
        Creates a logger with a unique name and path.
        """
        log_path = os.path.join(path, f'{name.lower().strip()}.log')
        
        # Create a unique logger name using the instance id
        logger_name = f'{name}_{self.times}'
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Create file and stream handlers
        file_handler = logging.FileHandler(log_path)
        stream_handler = logging.StreamHandler()
        
        # Set logging format
        formatter = logging.Formatter(f'%(asctime)s ~ %(levelname)s ~ %(lineno)-4.4d ~ {name} ~ %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        
        self.logger.info(f'Logger created at {log_path}')
    
    def set_configs(self, configs, **kwargs):
        """
        Sets the configuration variables from the configs dictionary.

        Args:
            configs: A dictionary of configuration arguments.
        """
        if isinstance(configs, Namespace):
            for key, value in vars(configs).items():
                setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.configs = configs
    
    def mkdir(self):
        self.save_path = os.path.join(self.save_path, str(self.times))
        self.model_path = os.path.join(self.save_path, 'models')
        self.model_info_path = os.path.join(self.save_path, 'models_info')
        self.log_path = os.path.join(self.save_path, 'logs')
        self.result_path = os.path.join(self.save_path, 'results')
        for dir in [
            self.save_path,
            self.model_path,
            self.log_path,
            self.model_info_path,
            self.result_path,
        ]:
            if not os.path.exists(dir):
                os.makedirs(dir)