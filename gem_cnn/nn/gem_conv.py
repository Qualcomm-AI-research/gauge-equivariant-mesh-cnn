# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
"""
Main convolution.

Data is arranged as:

x[number of vertices in bach, number of channels, dimensionality of representation]
"""
from functools import partial

import torch
from torch.nn import Parameter
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn.conv import MessagePassing

from torch_geometric.nn.inits import glorot, zeros

from gem_cnn.utils.einsum import einsum
from gem_cnn.utils.kernel import build_kernel
from gem_cnn.utils.rep_act import rep_act


class GemConv(MessagePassing):
    """
    GEM Convolution

    Args:
        in_channels (int): number of input features
        out_channels (int): number of output features
        in_order (int): order of input
        out_order (int): order of output
        n_rings (int): number of radial rings
        band_limit (int, optional): maximum theta frequency used
        batch (int, optional): compute edges in batches, checkpointed to save memory
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_order,
        out_order,
        n_rings,
        band_limit=None,
        batch=None,
    ):
        super().__init__(aggr="mean", flow="target_to_source", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_order = in_order
        self.out_order = out_order
        self.n_rings = n_rings
        # self.kernel has shape [n_bases, 2 * band_limit + 1, 2 * order_out + 1, 2 * order_in + 1]
        self.register_buffer(
            "kernel",
            torch.tensor(build_kernel(in_order, out_order, band_limit), dtype=torch.float32),
        )
        self.weight = Parameter(
            torch.Tensor(self.kernel.shape[0], n_rings, out_channels, in_channels)
        )
        self.register_buffer("bias_mask", torch.eye(2 * out_order + 1)[0])  # Only bias trivial rep
        self.bias = Parameter(torch.Tensor(out_channels))
        self.batch = batch

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, precomp, connection):
        assert x.shape[1] == self.in_channels
        assert x.shape[2] == 2 * self.in_order + 1
        assert precomp.dim() == 3
        out = self.propagate(edge_index=edge_index, x=x, precomp=precomp, connection=connection)

        return out

    def message(self, x_j, precomp, connection):
        """
        :param x_j: [n_edges, in_channels, 2*in_order+1]
        :param precomp [n_edges, 2*band_limit+1, n_rings]
        :param connection: [n_edges]
        :return: [num_v, out_channels, 2*out_order+1]
        """
        assert (
            self.kernel.shape[1] <= precomp.shape[1]
        ), "Kernel set with higher band-limit than precompute"
        precomp = precomp[:, : self.kernel.shape[1]]

        x_j_transported = rep_act(x_j, connection)
        if self.batch is None:
            y = einsum(
                "ejm,efr,bfnm,brij->ein",
                x_j_transported,
                precomp,
                self.kernel,
                self.weight,
            )
        else:
            ys = []
            for i in range(0, x_j.shape[0], self.batch):
                y = checkpoint(
                    partial(einsum, "ejm,efr,bfnm,brij->ein"),
                    x_j_transported[i : i + self.batch],
                    precomp[i : i + self.batch],
                    self.kernel,
                    self.weight,
                )
                ys.append(y)
            y = torch.cat(ys)
        y = y + self.bias[:, None] * self.bias_mask
        return y
