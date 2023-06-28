# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
"""
Main convolution.

Data is arranged as:

x[number of vertices in bach, number of channels, dimensionality of representation]
"""
import math

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import zeros

from gem_cnn.utils.rep_act import rep_act


def build_self_interaction_kernel(in_order, out_order):
    num_bases = min(in_order, out_order) * 2 + 1
    kernels = torch.zeros(num_bases, 2 * in_order + 1, 2 * out_order + 1)
    for i in range(min(in_order, out_order) + 1):
        if i == 0:
            kernels[i, 0, 0] = 1
        else:
            kernels[2 * i - 1, 2 * i - 1, 2 * i - 1] = 1
            kernels[2 * i - 1, 2 * i, 2 * i] = 1
            kernels[2 * i, 2 * i - 1, 2 * i] = 1
            kernels[2 * i, 2 * i, 2 * i - 1] = -1
    return kernels


class SelfInteractionWeight(nn.Module):
    def __init__(self, in_channels, out_channels, in_order, out_order):
        super().__init__()
        self.register_buffer("kernels", build_self_interaction_kernel(in_order, out_order))  # [bmn]
        self.weight = nn.Parameter(
            torch.empty((len(self.kernels), in_channels, out_channels))
        )  # [bji]
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.weight.shape[1] * self.kernels.shape[1]
        fan_out = self.weight.shape[2] * self.kernels.shape[2]
        stdv = math.sqrt(6.0 / (fan_in + fan_out))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self):
        """

        :return: [in_channels, 2*in_order+1, out_channels, 2*out_order+1]
        """
        return torch.einsum("bji,bmn->jmin", self.weight, self.kernels)


class GemConv(MessagePassing):
    """
    GEM Convolution

    Args:
        in_channels (int): number of input features
        out_channels (int): number of output features
        in_order (int): order of input
        out_order (int): order of output
        n_rings (int): number of radial rings
        band_limit (int, optional): maximum theta frequency used  DOES NOTHING CURRENTLY
        batch (int, optional): compute edges in batches, checkpointed to save memory DOES NOTHING CURRENTLY
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
        weight_shape = (n_rings, in_channels, 2 * in_order + 1, out_channels, 2 * out_order + 1)
        self.weight = nn.Parameter(torch.Tensor(*weight_shape))
        self.self_weight = SelfInteractionWeight(in_channels, out_channels, in_order, out_order)
        self.bias = nn.Parameter(torch.Tensor(out_channels, 2 * out_order + 1))
        self.register_buffer(
            "bias_mask", torch.eye(2 * out_order + 1)[0]
        )  # Only bias trivial rep on self interactions

        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.bias)
        fan_in = self.weight.shape[1] * self.weight.shape[2]
        fan_out = self.weight.shape[3] * self.weight.shape[4]
        stdv = math.sqrt(6.0 / (fan_in + fan_out))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, precomp, connection):
        assert x.shape[1] == self.in_channels
        assert x.shape[2] == 2 * self.in_order + 1
        out = self.propagate(edge_index=edge_index, x=x, precomp=precomp)

        return out

    def message(self, x_j, precomp):
        """
        :param x_j: [n_edges, in_channels, 2*in_order+1]
        :param precomp: [n_edges, 2+n_rings]
        :return: [num_v, out_channels, 2*out_order+1]
        """
        angle_pre, angle_post = precomp[:, :2].T
        radial_weights = precomp[:, 2:]
        x_j_transported = rep_act(x_j, angle_pre)  # connection-theta
        weight = torch.cat([self.self_weight()[None], self.weight])

        y_theta0 = torch.einsum(
            "ejm,er,rjmin->ein",
            x_j_transported,
            radial_weights,
            weight,
        )
        not_self = radial_weights[:, 0] == 0
        y_theta0 = y_theta0 + self.bias * not_self[:, None, None]
        y_theta0 = y_theta0 + self.bias * self.bias_mask * ~not_self[:, None, None]
        y = rep_act(y_theta0, angle_post)  # +theta
        return y
