from .base import BaseTopology

class Ring(BaseTopology):
    def generate_connections(self) -> dict:
        """
        Generate ring topology.
        
        Returns:
            A dictionary where keys are client IDs and values are lists of connected client IDs.
        """
        connections = {i: [(i - 1) % self.num_clients, (i + 1) % self.num_clients] for i in range(self.num_clients)}
        return connections