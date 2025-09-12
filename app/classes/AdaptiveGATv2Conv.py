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
        goal = (0.0, 0.0),  # Coordinate del punto fisso (beta, gamma)
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
                    index: Tensor, ptr: Optional[Tensor], dim_size: Optional[int],) -> Tensor:

        x_i_coords = x_i[..., -2:] # TODO Put the actual coordinates
        x_j_coords = x_j[..., -2:]

        pi = (x_i_coords - x_j_coords)[0]
        weight_1 = torch.exp(self.l * pi)

        distance = torch.sqrt(torch.sum((x_i_coords - self.goal) ** 2, dim=-1))
        weight_2 = torch.exp((1 - self.l) * distance)


        x = x_i + x_j
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)

        alpha = alpha * weight_1.unsqueeze(-1) * weight_2.unsqueeze(-1)

        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

