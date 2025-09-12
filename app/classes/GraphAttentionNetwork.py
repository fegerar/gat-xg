import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Dict, Any

from app.classes.AdaptiveGATv2Conv import AdaptiveGATv2Conv


class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network for Expected Goals prediction.

    This model processes whole graphs representing ball possession
    and predicts the expected goals (xG) value.
    """

    def __init__(
        self,
        input_features: int = 2,
        hidden_dim: int = 64,
        gat_layers: int = 2,
        gat_heads: int = 4,
        fc_hidden_dim: int = 128,
        dropout: float = 0.1,
        pool_method: str = 'mean'
    ):
        """Initialize Graph GAT model."""
        super(GraphAttentionNetwork, self).__init__()

        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.gat_layers = gat_layers
        self.gat_heads = gat_heads
        self.fc_hidden_dim = fc_hidden_dim
        self.dropout = dropout
        self.pool_method = pool_method

        # Input projection
        self.input_projection = nn.Linear(input_features, hidden_dim)

        # GAT layers - simplified to always produce consistent output
        self.gat_convs = nn.ModuleList()
        for i in range(gat_layers):
            # All layers have same input/output dimensions for simplicity
            gat_conv = AdaptiveGATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=gat_heads,
                concat=False,  # Always average heads for consistent dimensions
                dropout=dropout,
                add_self_loops=True,
                goal=(120.0, 40.0)  # Statsbomb definition
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

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, 1),
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

    def forward(self, batch_data: Dict[str, Any], return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.

        Args:
            batch_data: Dictionary containing 'graph' - Batched Data object
            return_attention: If True, also return per-layer attention weights.

        Returns:
            Dictionary containing 'predictions' - xG predictions [batch_size, 1]
            Optionally adds 'attentions': list of dicts with keys 'edge_index', 'alpha' (E,H) per layer
        """
        # Get the graph data
        data = batch_data['graph']
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Handle empty graphs
        if x.size(0) == 0:
            out = {
                'predictions': torch.zeros(1, 1, device=x.device if x.numel() > 0 else torch.device('cpu'))
            }
            if return_attention:
                out['attentions'] = []
            return out

        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)

        attentions = [] if return_attention else None

        # GAT layers
        for i, gat_conv in enumerate(self.gat_convs):
            if return_attention:
                x, (ei, alpha) = gat_conv(x, edge_index, return_attention_weights=True)
                attentions.append({
                    'layer': i,
                    'edge_index': ei.detach().cpu(),  # [2, E]
                    'alpha': alpha.detach().cpu()      # [E, H]
                })
            else:
                x = gat_conv(x, edge_index)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)

        # Graph pooling to get a single vector per graph
        graph_embedding = self.pool(x, batch)

        # Make predictions
        predictions = self.prediction_head(graph_embedding)

        result = {
            'predictions': predictions
        }
        if return_attention:
            result['attentions'] = attentions
        return result

    def predict(self, data: Data) -> torch.Tensor:
        """
        Make predictions on new data.

        Args:
            data: A single Data object

        Returns:
            xG prediction [1, 1]
        """
        self.eval()
        with torch.no_grad():
            batch_data = {'graph': Batch.from_data_list([data])}
            result = self.forward(batch_data)
            return result['predictions']
