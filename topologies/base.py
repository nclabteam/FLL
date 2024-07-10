class BaseTopology:
    def __init__(self, num_clients: int) -> None:
        """
        Initialize the Topology.

        Args:
            num_clients: Number of clients in the network.
        """
        self.num_clients = num_clients
        self.connections = self.generate_connections()

    def generate_connections(self) -> dict:
        """
        Generate connections between clients.
        Should be implemented by subclasses.
        
        Returns:
            A dictionary where keys are client IDs and values are lists of connected client IDs.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def display_connections(self) -> None:
        """
        Display the connections between clients.
        """
        for client, connected_clients in self.connections.items():
            print(f"Client {client} is connected to: {connected_clients}")
