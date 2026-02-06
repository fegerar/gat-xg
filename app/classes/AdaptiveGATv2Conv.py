from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.utils import softmax

from torch_geometric.nn import GATv2Conv


class AdaptiveGATv2Conv(GATv2Conv):

    def __init__(
            self,
            in_channels,
            out_channels,
            heads=1,
            concat=True,
            negative_slope=0.2,
            dropout=0.0,
            add_self_loops=True,
            bias=True,
            goal=(0.0, 0.0),  
            **kwargs,
    ):
        super(AdaptiveGATv2Conv, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            bias=bias,
            **kwargs
        )

        self.goal = torch.tensor(goal)
        self.l = Parameter(torch.tensor(0.6))

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Optional[Tensor],
                    index: Tensor, ptr: Optional[Tensor], dim_size: Optional[int], ) -> Tensor:
        # x_i, x_j expected shape: [E, H, F_out] (as in standard GATv2Conv implementation)
        # We interpret the first two feature dimensions (if available) of each head as coordinates.
        # If fewer than 2 output features, we skip adaptive weighting.

        # Standard GATv2 attention pre-activation (same as parent with x_i + x_j aggregation)
        x = x_i + x_j  # shape [E, H, F_out]
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)  # [E, H]

        # Coordinate-based adaptive weighting producing shape [E, H]
        if self.out_channels >= 2:
            # Extract (x, y) coordinates per head from transformed features
            # coords shape: [E, H, 2]
            coords_i = x_i[..., :2]
            coords_j = x_j[..., :2]

            # Relative displacement per edge/head -> take norm over coordinate dim -> [E, H]
            rel_disp = coords_j - coords_i  # [E, H, 2]
            rel_dist = torch.norm(rel_disp, dim=-1)  # [E, H]
            # Distance to goal per receiving node (goal broadcast):
            goal = self.goal.to(alpha.device).view(1, 1, 2)  # [1,1,2]
            dist_to_goal = torch.norm(coords_j - goal, dim=-1)  # [E, H]

            # Scalar weights per head
            weight_1 = torch.exp(self.l * rel_dist)          # [E, H]
            weight_2 = torch.exp(-(1 - self.l) * dist_to_goal) # [E, H]

            alpha = alpha * weight_1 * weight_2
        # else: keep original alpha (cannot derive 2D coordinates)

        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha