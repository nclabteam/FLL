from .base import BaseServer, BaseClient

class LocalOnly_Server(BaseServer):
    def server_aggregation(self):
        pass

    def send_models(self):
        for client in self.clients:
            client.metrics['send_time'].append(0)

    def receive_models(self):
        pass

class LocalOnly_Client(BaseClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save = True
        
    def initialize_local(self, *args, **kwargs):
        pass