# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
"""
Adapted from: hsn/nn/harmonic_resnet_block.py by Ruben Wiersma at github.com/rubenwiersma/hsn

MIT License

Copyright (c) 2020 rubenwiersma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from gem_cnn.nn.gem_conv import GemConv
from gem_cnn.nn.regular_nonlin import RegularNonlinearity


class GemResNetBlock(torch.nn.Module):
    """
    ResNet block with convolutions, linearities, and non-linearities

    Args:
        in_channels (int): number of input features
        out_channels (int): number of output features
        in_order (int): order of input
        out_order (int): order of output
        n_rings (int): number of radial rings
        num_samples (int): number of samples to use for non-linearity. Should be odd
        band_limit (int, optional): maximum theta frequency used
        last_layer (bool): whether to apply final non-linearity
        checkpoint (bool): whether to call GemConv within a torch checkpoint, saving lots of memory
        batch_norm (bool): whether use batch norm before non-lienarities
        batch (int, optional): if not None, comptue conv in batches of this size, checkpointed
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_order,
        out_order,
        n_rings,
        num_samples,
        band_limit=None,
        last_layer=False,
        checkpoint=False,
        batch_norm=False,
        batch=None,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        middle_order = max(in_order, out_order)
        self.conv1 = GemConv(
            in_channels,
            out_channels,
            in_order,
            middle_order,
            n_rings,
            band_limit,
            batch,
        )
        self.conv2 = GemConv(
            out_channels,
            out_channels,
            middle_order,
            out_order,
            n_rings,
            band_limit,
            batch,
        )

        # Apply batch norm inside RegularNonLinearity
        if batch_norm:
            act1 = nn.Sequential(nn.BatchNorm1d(out_channels), nn.ReLU())
            act2 = nn.Sequential(nn.BatchNorm1d(out_channels), nn.ReLU())
        else:
            act1 = act2 = nn.ReLU()

        self.nonlin1 = RegularNonlinearity(middle_order, num_samples, act1)

        if last_layer:
            self.nonlin2 = nn.Identity()
        else:
            self.nonlin2 = RegularNonlinearity(out_order, num_samples, act2)

        if in_channels != out_channels:
            self.lin = nn.Conv1d(
                in_channels, out_channels, kernel_size=1
            )  # Equivariant linear map mixing channels
        else:
            self.lin = nn.Identity()

    @staticmethod
    def call_conv_dummy(conv, x, edge_index, precomp, connection, _dummy):
        return conv(x, edge_index, precomp, connection)

    def call_conv(self, conv, x, edge_index, precomp, connection):
        if self.checkpoint:
            # Create dummy requires_grad argument to suppress pytorch checkpoint warning
            dummy = torch.zeros(1, device=x.device).requires_grad_()
            return checkpoint(
                partial(self.call_conv_dummy, conv),
                x,
                edge_index,
                precomp,
                connection,
                dummy,
            )
        return conv(x, edge_index, precomp, connection)

    def add_residual(self, y, x):
        residual = self.lin(x)
        o = min(y.shape[2], residual.shape[2])
        y[:, :, :o] = y[:, :, :o] + residual[:, :, :o]  # Handle varying orders
        return y

    def forward(self, x, edge_index, precomp, connection):
        """
        Forward pass.

        :param x: [num_v, in_channels, 2*in_order+1]
        :param edge_index: [n_edges, 2]
        :param precomp: [n_edges, 2*band_limit+1, n_rings] computed by GemPrecomp
        :param connection: [num_edges]
        :return: [num_v, out_channels, 2*out_order+1]
        """
        y = self.call_conv(self.conv1, x, edge_index, precomp, connection)
        y = self.nonlin1(y)
        y = self.call_conv(self.conv2, y, edge_index, precomp, connection)
        y = self.add_residual(y, x)

        return self.nonlin2(y)
