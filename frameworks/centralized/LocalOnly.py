from .base import Server, Client

class LocalOnly(Server):
    def server_aggregation(self):
        pass

    def send_models(self):
        for client in self.clients:
            client.metrics['send_time'].append(0)

    def receive_models(self):
        pass

class LocalOnly_Client(Client):
    pass
