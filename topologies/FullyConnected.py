from .base import BaseTopology

class FullyConnected(BaseTopology):
    def generate_connections(self) -> dict:
        """
        Generate fully connected topology.
        
        Returns:
            A dictionary where keys are client IDs and values are lists of connected client IDs.
        """
        connections = {i: [j for j in range(self.num_clients) if j != i] for i in range(self.num_clients)}
        return connections