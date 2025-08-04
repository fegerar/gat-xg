import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import List, Optional, Dict, Any, Union


class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer implementation.
    
    This layer performs attention-based message passing on graph-structured data,
    allowing nodes to attend to their neighbors with learned attention weights.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        use_gatv2: bool = False
    ):
        """
        Initialize Graph Attention Layer.
        
        Args:
            in_features: Size of input node features
            out_features: Size of output node features
            heads: Number of attention heads
            concat: Whether to concatenate or average multi-head outputs
            negative_slope: LeakyReLU negative slope for attention coefficients
            dropout: Dropout probability for attention coefficients
            add_self_loops: Whether to add self-loops to the graph
            bias: Whether to use bias in linear transformations
            use_gatv2: Whether to use GATv2 (improved) instead of standard GAT
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.use_gatv2 = use_gatv2
        
        # Choose GAT variant
        if use_gatv2:
            self.conv = GATv2Conv(
                in_channels=in_features,
                out_channels=out_features,
                heads=heads,
                concat=concat,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=add_self_loops,
                bias=bias
            )
        else:
            self.conv = GATConv(
                in_channels=in_features,
                out_channels=out_features,
                heads=heads,
                concat=concat,
                negative_slope=negative_slope,
                dropout=dropout,
                add_self_loops=add_self_loops,
                bias=bias
            )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                return_attention_weights: bool = False) -> Union[torch.Tensor, tuple]:
        """
        Forward pass through the attention layer.
        
        Args:
            x: Node feature matrix [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Node embeddings [num_nodes, out_features * heads] if concat=True
            or [num_nodes, out_features] if concat=False
            Optionally returns attention weights if requested
        """
        if return_attention_weights:
            return self.conv(x, edge_index, return_attention_weights=True)
        else:
            return self.conv(x, edge_index)


class ProgressiveGAT(nn.Module):
    """
    Simplified Progressive Graph Attention Network for Expected Goals prediction.
    
    This model processes sequences of graphs representing ball possession sequences
    and predicts the expected goals (xG) value.
    """
    
    def __init__(
        self,
        input_features: int = 2,
        hidden_dim: int = 64,
        gat_layers: int = 2,
        gat_heads: int = 4,
        sequence_hidden_dim: int = 128,
        dropout: float = 0.1,
        pool_method: str = 'mean'
    ):
        """Initialize Progressive GAT model."""
        super(ProgressiveGAT, self).__init__()
        
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.gat_layers = gat_layers
        self.gat_heads = gat_heads
        self.sequence_hidden_dim = sequence_hidden_dim
        self.dropout = dropout
        self.pool_method = pool_method
        
        # Input projection
        self.input_projection = nn.Linear(input_features, hidden_dim)
        
        # GAT layers - simplified to always produce consistent output
        self.gat_convs = nn.ModuleList()
        for i in range(gat_layers):
            # All layers have same input/output dimensions for simplicity
            gat_conv = GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=gat_heads,
                concat=False,  # Always average heads for consistent dimensions
                dropout=dropout,
                add_self_loops=True
            )
            self.gat_convs.append(gat_conv)
        
        # Layer normalization for each GAT layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(gat_layers)
        ])
        
        # Pooling
        if pool_method == 'mean':
            self.pool = global_mean_pool
        elif pool_method == 'max':
            self.pool = global_max_pool
        elif pool_method == 'add':
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pool_method}")
        
        # Sequence processing
        self.sequence_processor = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=sequence_hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
            bidirectional=False
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(sequence_hidden_dim, sequence_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sequence_hidden_dim // 2, sequence_hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(sequence_hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward_single_graph(self, data: Data) -> torch.Tensor:
        """
        Process a single graph through GAT layers.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Graph embedding tensor of shape [1, hidden_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        
        # Handle empty graphs
        if x.size(0) == 0:
            return torch.zeros(1, self.hidden_dim, device=x.device if x.numel() > 0 else torch.device('cpu'))
        
        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        
        # GAT layers
        for i, gat_conv in enumerate(self.gat_convs):
            x = gat_conv(x, edge_index)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # Graph pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        graph_embedding = self.pool(x, batch)
        
        # Ensure output has correct shape [1, hidden_dim]
        if graph_embedding.dim() == 1:
            graph_embedding = graph_embedding.unsqueeze(0)
        
        return graph_embedding
    
    def forward(self, batch_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            batch_data: Dictionary containing 'sequences' - List of lists of Data objects
                
        Returns:
            Dictionary containing 'predictions' - xG predictions [batch_size, 1]
        """
        sequences = batch_data['sequences']
        batch_size = len(sequences)
        
        # Process each sequence
        all_sequence_embeddings = []
        
        for sequence in sequences:
            # Process each graph in the sequence
            graph_embeddings = []
            for graph in sequence:
                graph_embedding = self.forward_single_graph(graph)
                # Ensure consistent shape [hidden_dim]
                if graph_embedding.dim() > 1:
                    graph_embedding = graph_embedding.squeeze(0)
                graph_embeddings.append(graph_embedding)
            
            # Stack to create sequence tensor [seq_len, hidden_dim]
            sequence_tensor = torch.stack(graph_embeddings)
            all_sequence_embeddings.append(sequence_tensor)
        
        # Pad sequences to same length
        max_len = max(seq.size(0) for seq in all_sequence_embeddings)
        padded_sequences = torch.zeros(batch_size, max_len, self.hidden_dim, 
                                     device=all_sequence_embeddings[0].device)
        
        for i, seq in enumerate(all_sequence_embeddings):
            padded_sequences[i, :seq.size(0)] = seq
        
        # Process with LSTM
        lstm_output, (hidden, _) = self.sequence_processor(padded_sequences)
        
        # Use last hidden state
        sequence_representation = hidden[-1]  # [batch_size, sequence_hidden_dim]
        
        # Make predictions
        predictions = self.prediction_head(sequence_representation)
        
        return {
            'predictions': predictions
        }
    
    def predict(self, sequences: List[List[Data]]) -> torch.Tensor:
        """
        Make predictions on new data.
        
        Args:
            sequences: List of sequences, where each sequence is a list of Data objects
            
        Returns:
            xG predictions [batch_size, 1]
        """
        self.eval()
        with torch.no_grad():
            batch_data = {'sequences': sequences}
            result = self.forward(batch_data)
            return result['predictions']
