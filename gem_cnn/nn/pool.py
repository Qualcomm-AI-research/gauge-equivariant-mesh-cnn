# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import torch
from torch_geometric.nn import MessagePassing

from gem_cnn.transform.scale_mask import mask_idx, invert_index
from gem_cnn.utils.rep_act import rep_act


class ParallelTransportPool(MessagePassing):
    r"""
    Pooling layer with parallel transport

    Args:
        coarse_lvl (int): scale to pool to
        unpool (bool): whether to do unpooling
    """

    def __init__(self, coarse_lvl, *, unpool):
        super().__init__(aggr="mean", flow="target_to_source", node_dim=0)
        self.coarse_lvl = coarse_lvl
        self.unpool = unpool

    def forward(self, x, data):
        pool_edge_mask = mask_idx(2 * self.coarse_lvl, data.edge_mask)
        node_idx_fine = torch.nonzero(data.node_mask >= self.coarse_lvl - 1).view(-1)
        node_idx_coarse = torch.nonzero(data.node_mask >= self.coarse_lvl).view(-1)
        node_idx_all_to_fine = invert_index(node_idx_fine, data.num_nodes)
        node_idx_all_to_coarse = invert_index(node_idx_coarse, data.num_nodes)

        coarse, fine = data.edge_index[:, pool_edge_mask]
        coarse_idx_coarse = node_idx_all_to_coarse[coarse]
        fine_idx_fine = node_idx_all_to_fine[fine]

        num_fine, num_coarse = node_idx_fine.shape[0], node_idx_coarse.shape[0]

        if self.unpool:
            connection = -data.connection[pool_edge_mask]  # Parallel transport inverse
            edge_index = torch.stack([fine_idx_fine, coarse_idx_coarse])  # Coarse to fine
            size = (num_coarse, num_fine)
        else:  # Pool
            connection = data.connection[pool_edge_mask]  # Parallel transport
            edge_index = torch.stack([coarse_idx_coarse, fine_idx_fine])  # Fine to coarse
            size = (num_fine, num_coarse)

        out = self.propagate(edge_index=edge_index, x=x, connection=connection, size=size)
        return out

    def message(self, x_j, connection):
        """
        Applies connection to each neighbour, before aggregating for pooling.
        """
        x_j_transported = rep_act(x_j, connection)
        return x_j_transported
