"""
Classes module for Progressive Graph Attention Networks (ProGAT-XG).

This module contains the core model architectures and dataset classes for
Expected Goals (xG) prediction using Graph Attention Networks on soccer data.
"""

from .ProgressiveGraphSoccerDataset import ProgressiveGraphSoccerDataset
from .ProgressiveGAT import ProgressiveGAT, GraphAttentionLayer

__all__ = [
    'ProgressiveGraphSoccerDataset',
    'ProgressiveGAT',
    'GraphAttentionLayer', 

]
