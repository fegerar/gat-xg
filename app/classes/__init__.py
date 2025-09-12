"""
Classes module for Graph Attention Networks (GAT-XG).

This module contains the core model architectures and dataset classes for
Expected Goals (xG) prediction using Graph Attention Networks on soccer data.
"""

from .GraphSoccerDataset import GraphSoccerDataset
from .GraphAttentionNetwork import GraphAttentionNetwork
from .AdaptiveGATv2Conv import AdaptiveGATv2Conv

__all__ = [
    'GraphSoccerDataset',
    'GraphAttentionNetwork',
    'AdaptiveGATv2Conv'
]
