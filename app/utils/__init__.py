# Utils package for progat-xg
# 
# This package contains utilities for processing StatsBomb data and creating visualizations
# for progressive graph analysis in football/soccer data.

from .dataset import (
    download_github_directory,
    get_event_by_id,
    split_ball_possessions,
    select_only_shot_possessions,
    possession_to_graph,
    progressive_graphs,
    game2graphs,  # Backward compatibility wrapper
    explore_possessions as dataset_explore_possessions  # Backward compatibility wrapper
)

__all__ = [
    # Dataset functions
    'download_github_directory',
    'get_event_by_id',
    'split_ball_possessions',
    'select_only_shot_possessions',
    'possession_to_graph',
    'progressive_graphs',
    'game2graphs',  # Backward compatibility
    'dataset_explore_possessions',  # Backward compatibility
]
