# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
"""
Adapted from: hsn/transforms/multiscale_radius_graph.py by Ruben Wiersma at github.com/rubenwiersma/hsn

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

import torch

from torch_sparse import coalesce
from torch_geometric.nn import radius, fps, knn


class MultiscaleRadiusGraph:
    r"""Creates a radius graph for multiple pooling levels.
    The nodes and adjacency matrix for each pooling level can be accessed by masking
    tensors with values for nodes and edges with data.node_mask and data.edge_mask, respectively.

    Edges can belong to multiple levels,
    therefore we store the membership of an edge for a certain level with a bitmask:
        - The bit at position 2 * n corresponds to the edges used for pooling to level n
        - The bit at position 2 * n + 1 corresponds to the edges used for convolution in level n

    To find out if an edge belongs to a level, use a bitwise AND:
        `edge_mask & (0b1 << lvl) > 0`

    Args:
        ratios (list): the ratios for downsampling at each pooling layer.
        radii (list): the radius of the kernel support for each scale.
        max_neighbours (int, optional): the maximum number of neighbors per vertex,
            important to set higher than the expected number of neighbors.
    """

    def __init__(self, ratios, radii, max_neighbours=512):
        assert len(ratios) == len(radii)
        self.ratios = ratios
        self.radii = radii
        self.max_neighbours = max_neighbours

    def __call__(self, data):
        data.edge_coords = None
        batch = data.batch if "batch" in data else None
        pos = data.pos

        # Create empty tensors to store edge indices and masks
        edge_index = []
        edge_mask = []
        node_mask = torch.zeros(data.num_nodes)

        # Sample points on the surface using farthest point sampling if sample_n is given
        original_idx = torch.arange(data.num_nodes)
        batch = batch if batch is not None else torch.zeros(data.num_nodes, dtype=torch.long)
        for i, r in enumerate(self.ratios):
            # POOLING EDGES
            # Sample a number of points given by ratio r
            # and create edges to sampled points from nearest neighbors
            if r == 1:
                pool_idx = original_idx
            else:
                pool_idx = fps(pos, batch, r).sort()[0]
                pool_neigh = knn(
                    x=pos[pool_idx], y=pos, k=1, batch_x=batch[pool_idx], batch_y=batch
                )[1]
                # Add edges for pooling
                edge_index.append(
                    torch.stack((original_idx[pool_idx][pool_neigh], original_idx), dim=0)
                )
                edge_mask.append(
                    torch.ones(original_idx.shape[0], dtype=torch.long) * (0b1 << (i * 2))
                )

            # Sample nodes
            original_idx = original_idx[pool_idx]
            pos = pos[pool_idx]
            batch = batch[pool_idx]
            node_mask[original_idx] = i

            # CONVOLUTION EDGES
            # Create a radius graph for pooled points
            radius_edges = radius(pos, pos, self.radii[i], batch, batch, self.max_neighbours)
            radius_edges = original_idx[radius_edges]
            edge_index.append(radius_edges)
            edge_mask.append(
                torch.ones(radius_edges.size(1), dtype=torch.long) * (0b1 << (i * 2 + 1))
            )

        # Sort edges and combine duplicates with an add (=bitwise OR, as bitwise & gives 0) operation
        edge_index = torch.cat(edge_index, dim=1)
        edge_mask = torch.cat(edge_mask)
        edge_index, edge_mask = coalesce(
            edge_index, edge_mask, data.num_nodes, data.num_nodes, "add"
        )

        # Store in data object
        data.edge_index = edge_index
        data.node_mask = node_mask
        data.edge_mask = edge_mask
        return data

    def __repr__(self):
        return "{}(radii={}, ratios={})".format(self.__class__.__name__, self.radii, self.ratios)
