import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any, Tuple


class GraphSoccerDataset(Dataset):
    """
    PyTorch Dataset for soccer possessions with whole graphs.

    Each sample contains:
    - graph: A single torch_geometric.data.Data object representing the whole possession
    - xg: Expected goals value (float)
    - game_id: Identifier for the game (string)
    """

    def __init__(
        self,
        pickle_path: str,
        normalize_coordinates: bool = True,
        field_dimensions: Tuple[float, float] = (120.0, 80.0),
        final_graph_only: bool = False
    ):
        """
        Initialize the dataset.

        Args:
            pickle_path: Path to the pickle file containing processed data
            normalize_coordinates: Whether to normalize coordinates to [0, 1]
            field_dimensions: Football field dimensions (length, width) for normalization
            final_graph_only: If True, use only the final graph in each possession sequence
        """
        self.pickle_path = pickle_path
        self.normalize_coordinates = normalize_coordinates
        self.field_dimensions = field_dimensions
        self.final_graph_only = final_graph_only

        # Load the data
        with open(pickle_path, 'rb') as f:
            raw_data = pickle.load(f)

        # Process data to extract the graphs we need
        self.data = self._process_raw_data(raw_data)

        print(f"Loaded {len(self.data)} graph samples from {pickle_path}")

        # Calculate statistics
        self._calculate_stats()

    def _process_raw_data(self, raw_data):
        """
        Process raw data to extract whole graphs.

        Args:
            raw_data: List of possession sequences

        Returns:
            List of processed data items with single graphs
        """
        processed_data = []

        for item in raw_data:
            if self.final_graph_only:
                # Take only the final graph in the possession
                if len(item['possession']) > 0:
                    graph = item['possession'][-1]  # Get the last graph
                    processed_data.append({
                        'graph': graph,
                        'xg': item['xg'],
                        'game_id': item['game_id'],
                    })
            else:
                # Create a new merged graph by combining all nodes and edges
                if len(item['possession']) > 0:
                    # Create a single graph by merging all the progressive graphs
                    merged_graph = self._merge_graphs(item['possession'])
                    processed_data.append({
                        'graph': merged_graph,
                        'xg': item['xg'],
                        'game_id': item['game_id'],
                    })

        return processed_data

    def _merge_graphs(self, graphs: List[Data]) -> Data:
        """
        Merge multiple graphs into a single graph.

        Args:
            graphs: List of Data objects

        Returns:
            A single Data object with combined nodes and edges
        """
        if not graphs:
            return Data(
                x=torch.empty((0, 2), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long)
            )

        # Simple approach: use the last graph in the sequence
        # This could be enhanced to merge all graphs in a more sophisticated way
        final_graph = graphs[-1]

        return final_graph

    def _calculate_stats(self):
        """Calculate dataset statistics."""
        self.xg_values = [item['xg'] for item in self.data]

        print(f"XG stats - Min: {min(self.xg_values):.4f}, Max: {max(self.xg_values):.4f}, Mean: {sum(self.xg_values)/len(self.xg_values):.4f}")

        # Calculate node and edge statistics
        node_counts = [item['graph'].x.size(0) for item in self.data]
        edge_counts = [item['graph'].edge_index.size(1) for item in self.data]

        print(f"Node count stats - Min: {min(node_counts)}, Max: {max(node_counts)}, Mean: {sum(node_counts)/len(node_counts):.2f}")
        print(f"Edge count stats - Min: {min(edge_counts)}, Max: {max(edge_counts)}, Mean: {sum(edge_counts)/len(edge_counts):.2f}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
            - graph: A Data object
            - xg: Expected goals value
            - game_id: Game identifier
        """
        item = self.data[idx]
        graph = item['graph']

        # Normalize coordinates if requested
        if self.normalize_coordinates:
            graph = self._normalize_graph(graph)

        return {
            'graph': graph,
            'xg': torch.tensor(item['xg'], dtype=torch.float32),
            'game_id': item['game_id']
        }

    def _normalize_graph(self, graph: Data) -> Data:
        """
        Normalize graph coordinates to [0, 1] based on field dimensions.

        Args:
            graph: Data object

        Returns:
            Data object with normalized coordinates
        """
        if graph.x.size(0) > 0:  # Only if there are nodes
            normalized_x = graph.x.clone()
            # Normalize x-coordinate (field length)
            normalized_x[:, 0] = normalized_x[:, 0] / self.field_dimensions[0]
            # Normalize y-coordinate (field width)
            normalized_x[:, 1] = normalized_x[:, 1] / self.field_dimensions[1]

            # Create new Data object with normalized coordinates
            normalized_graph = Data(
                x=normalized_x,
                edge_index=graph.edge_index,
                **{k: v for k, v in graph.items() if k not in ['x', 'edge_index']}
            )
            return normalized_graph
        else:
            return graph

    def get_samples_by_game_id(self, game_id: str) -> List[Dict[str, Any]]:
        """
        Get all samples from a specific game.

        Args:
            game_id: Game identifier

        Returns:
            List of samples from the specified game
        """
        return [item for item in self.data if item['game_id'] == game_id]

    def filter_by_xg_threshold(self, min_xg: float = 0.0, max_xg: float = 1.0) -> 'GraphSoccerDataset':
        """
        Create a new dataset filtered by xG threshold.

        Args:
            min_xg: Minimum xG value
            max_xg: Maximum xG value

        Returns:
            New GraphSoccerDataset with filtered data
        """
        filtered_data = [
            item for item in self.data
            if min_xg <= item['xg'] <= max_xg
        ]

        # Create new dataset with filtered data
        new_dataset = GraphSoccerDataset.__new__(GraphSoccerDataset)
        new_dataset.pickle_path = self.pickle_path
        new_dataset.normalize_coordinates = self.normalize_coordinates
        new_dataset.field_dimensions = self.field_dimensions
        new_dataset.final_graph_only = self.final_graph_only
        new_dataset.data = filtered_data

        new_dataset._calculate_stats()
        return new_dataset

    @staticmethod
    def collate_soccer_graphs(batch):
        """
        Custom collate function for batching graph data.

        Args:
            batch: List of sample dictionaries

        Returns:
            Batched data with:
            - graph: Batched Data object
            - xg: Tensor of xG values
            - game_id: List of game IDs
        """
        graphs = [item['graph'] for item in batch]
        xg_values = torch.stack([item['xg'] for item in batch])
        game_ids = [item['game_id'] for item in batch]

        # Batch graphs
        batched_graphs = Batch.from_data_list(graphs)

        return {
            'graph': batched_graphs,
            'xg': xg_values,
            'game_ids': game_ids
        }
