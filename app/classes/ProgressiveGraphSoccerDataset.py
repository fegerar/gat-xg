import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from typing import List, Dict, Any, Optional, Tuple
import random


class ProgressiveGraphSoccerDataset(Dataset):
    """
    PyTorch Dataset for soccer possession sequences with progressive graphs.
    
    Each sample contains:
    - possession: List of torch_geometric.data.Data objects representing progressive graphs
    - xg: Expected goals value (float)
    - game_id: Identifier for the game (string)
    """
    
    def __init__(
        self, 
        pickle_path: str,
        sequence_length: Optional[int] = None,
        use_full_sequence: bool = True,
        normalize_coordinates: bool = True,
        field_dimensions: Tuple[float, float] = (120.0, 80.0)
    ):
        """
        Initialize the dataset.
        
        Args:
            pickle_path: Path to the pickle file containing processed data
            sequence_length: If specified, truncate/pad sequences to this length
            use_full_sequence: If True, return the full possession sequence. 
                              If False, return a random graph from the sequence
            normalize_coordinates: Whether to normalize coordinates to [0, 1]
            field_dimensions: Football field dimensions (length, width) for normalization
        """
        self.pickle_path = pickle_path
        self.sequence_length = sequence_length
        self.use_full_sequence = use_full_sequence
        self.normalize_coordinates = normalize_coordinates
        self.field_dimensions = field_dimensions
        
        # Load the data
        with open(pickle_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data)} possession sequences from {pickle_path}")
        
        # Calculate statistics
        self._calculate_stats()
    
    def _calculate_stats(self):
        """Calculate dataset statistics."""
        self.xg_values = [item['xg'] for item in self.data]
        self.sequence_lengths = [len(item['possession']) for item in self.data]
        
        print(f"XG stats - Min: {min(self.xg_values):.4f}, Max: {max(self.xg_values):.4f}, Mean: {sum(self.xg_values)/len(self.xg_values):.4f}")
        print(f"Sequence length stats - Min: {min(self.sequence_lengths)}, Max: {max(self.sequence_lengths)}, Mean: {sum(self.sequence_lengths)/len(self.sequence_lengths):.2f}")
    
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
            - graphs: Either a list of Data objects or a single Data object
            - xg: Expected goals value
            - game_id: Game identifier
            - sequence_length: Original sequence length
        """
        item = self.data[idx]
        possession_graphs = item['possession'].copy()
        
        # Normalize coordinates if requested
        if self.normalize_coordinates:
            possession_graphs = self._normalize_graphs(possession_graphs)
        
        if self.use_full_sequence:
            # Return the full sequence (potentially truncated/padded)
            if self.sequence_length is not None:
                possession_graphs = self._process_sequence_length(possession_graphs)
            
            return {
                'graphs': possession_graphs,
                'xg': torch.tensor(item['xg'], dtype=torch.float32),
                'game_id': item['game_id'],
                'sequence_length': len(item['possession'])
            }
        else:
            # Return a random graph from the sequence
            random_idx = random.randint(0, len(possession_graphs) - 1)
            selected_graph = possession_graphs[random_idx]
            
            return {
                'graph': selected_graph,
                'xg': torch.tensor(item['xg'], dtype=torch.float32),
                'game_id': item['game_id'],
                'graph_position': random_idx,
                'sequence_length': len(possession_graphs)
            }
    
    def _normalize_graphs(self, graphs: List[Data]) -> List[Data]:
        """
        Normalize graph coordinates to [0, 1] based on field dimensions.
        
        Args:
            graphs: List of Data objects
            
        Returns:
            List of Data objects with normalized coordinates
        """
        normalized_graphs = []
        
        for graph in graphs:
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
                normalized_graphs.append(normalized_graph)
            else:
                normalized_graphs.append(graph)
        
        return normalized_graphs
    
    def _process_sequence_length(self, graphs: List[Data]) -> List[Data]:
        """
        Truncate or pad sequence to specified length.
        
        Args:
            graphs: List of Data objects
            
        Returns:
            List of Data objects with specified length
        """
        if len(graphs) > self.sequence_length:
            # Truncate
            return graphs[:self.sequence_length]
        elif len(graphs) < self.sequence_length:
            # Pad with empty graphs
            padding_needed = self.sequence_length - len(graphs)
            empty_graph = Data(
                x=torch.empty((0, 2), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long)
            )
            return graphs + [empty_graph] * padding_needed
        else:
            return graphs
    
    def get_sequence_by_game_id(self, game_id: str) -> List[Dict[str, Any]]:
        """
        Get all sequences from a specific game.
        
        Args:
            game_id: Game identifier
            
        Returns:
            List of sequences from the specified game
        """
        return [item for item in self.data if item['game_id'] == game_id]
    
    def filter_by_xg_threshold(self, min_xg: float = 0.0, max_xg: float = 1.0) -> 'ProgressiveGraphSoccerDataset':
        """
        Create a new dataset filtered by xG threshold.
        
        Args:
            min_xg: Minimum xG value
            max_xg: Maximum xG value
            
        Returns:
            New ProgressiveGraphSoccerDataset with filtered data
        """
        filtered_data = [
            item for item in self.data 
            if min_xg <= item['xg'] <= max_xg
        ]
        
        # Create new dataset with filtered data
        new_dataset = ProgressiveGraphSoccerDataset.__new__(ProgressiveGraphSoccerDataset)
        new_dataset.pickle_path = self.pickle_path
        new_dataset.sequence_length = self.sequence_length
        new_dataset.use_full_sequence = self.use_full_sequence
        new_dataset.normalize_coordinates = self.normalize_coordinates
        new_dataset.field_dimensions = self.field_dimensions
        new_dataset.data = filtered_data
        new_dataset._calculate_stats()
        
        return new_dataset
    
    @staticmethod
    def collate_football_sequences(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for batching football sequences.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Batched data ready for model training
        """
        if 'graphs' in batch[0]:  # Full sequence mode
            # Batch sequences
            batched_sequences = []
            max_seq_len = max(len(sample['graphs']) for sample in batch)
            
            for sample in batch:
                graphs = sample['graphs']
                # Pad sequence if necessary
                while len(graphs) < max_seq_len:
                    empty_graph = Data(
                        x=torch.empty((0, 2), dtype=torch.float32),
                        edge_index=torch.empty((2, 0), dtype=torch.long)
                    )
                    graphs.append(empty_graph)
                batched_sequences.append(graphs)
            
            return {
                'sequences': batched_sequences,
                'xg': torch.stack([sample['xg'] for sample in batch]),
                'game_ids': [sample['game_id'] for sample in batch],
                'sequence_lengths': torch.tensor([sample['sequence_length'] for sample in batch])
            }
        
        else:  # Single graph mode
            # Batch individual graphs
            graphs = [sample['graph'] for sample in batch]
            batched_graphs = Batch.from_data_list(graphs)
            
            return {
                'graphs': batched_graphs,
                'xg': torch.stack([sample['xg'] for sample in batch]),
                'game_ids': [sample['game_id'] for sample in batch],
                'graph_positions': torch.tensor([sample['graph_position'] for sample in batch]),
                'sequence_lengths': torch.tensor([sample['sequence_length'] for sample in batch])
            }



