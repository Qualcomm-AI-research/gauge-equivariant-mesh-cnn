# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import numpy as np
import torch
from torch_scatter import scatter


def so3_matrix_generator(axis, theta):
    """
    SO(3) exponential map, vectorized.
    By Rodriguez formula.
    :param axis: in S2 rotation axis (B, 3)
    :param theta: Rotation angle (B,)
    :return: Rotation matrix in SO(3), (B, 3, 3)
    """
    theta = np.asarray(theta)

    theta = theta[:, None, None]
    x, y, z = axis.T
    zero = np.zeros_like(x)
    k = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], 1).reshape((-1, 3, 3))
    rot = np.eye(3)[None] + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k

    return rot


def spherical_parallel_transport(p_from, p_to, v):
    """
    Parallel transport according to Levi-Civita connection on sphere along geodesic.
    Vectorized.
    :param p_from: in S2, (B, 3)
    :param p_to: in S2, (B, 3)
    :param v: in T_{p_from} S2 (B, 3)
    :return: v_transformed in T_{p_to} S2 (B, 3)
    """
    assert p_from.shape == p_to.shape == v.shape
    axis = np.cross(p_from, p_to)
    axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-20)
    theta = np.arccos(np.sum(p_to * p_from, axis=1).clip(-1, 1))
    rot = so3_matrix_generator(axis, theta)
    v_transformed = np.einsum("nij,nj->ni", rot, v)
    return v_transformed


def find_first_neighbour(edge_index):
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]  # Ignore self loops
    num_e = edge_index.shape[1]
    ar = torch.arange(num_e, device=edge_index.device)
    first_neighbour_idx = scatter(ar, index=edge_index[0], reduce="min")
    first_neighbour = edge_index[1, first_neighbour_idx]
    return first_neighbour


class SimpleGeometry:
    """
    Computes logarithmic map by projection and parallel transporters using spherical approximation.

    It just requires a normal vector for each vertex.
    The log-map follows from from projecting the difference 3-D vector to the tangent plane.
    The transporters are then defined as the those from spherical geometry with the corresponding normal vectors.

    It assumes the data object is annotated with:
    - normal [num_vertices, 3]

    It annotates the data object with the tensor attributes:
    - edge_coords: [num_edges, 2]. the log-map in polar coords
    - connection: [num_edges]. the parallel transporter (q->p) as an angle
    - weight: [num_edges]. the quadrature weight for the neighbour
    """

    def __init__(self, gauge_def="first_neighbour"):
        """

        :param gauge_def: choice [random, first_neighbour, x]
        """
        self.gauge_def = gauge_def

    def gauge_defining_vector(self, data, device, dtype):
        num_v = data.normal.shape[0]
        if self.gauge_def == "random":
            return torch.randn(num_v, 3, device=device, dtype=dtype)
        elif self.gauge_def == "first_neighbour":
            first_neighbour = find_first_neighbour(data.edge_index)
            return data.pos[first_neighbour] - data.pos
        elif self.gauge_def == "x":
            return torch.tensor([1, 0, 0], device=device, dtype=dtype).expand_as(data.pos)
        else:
            raise ValueError()

    def __call__(self, data):
        """
        Perform annotation
        :param data: Data object
        :return: Data object
        """
        opts = dict(device=data.pos.device, dtype=data.pos.dtype)
        normal = data.normal / data.normal.norm(p=2, dim=1, keepdim=True)
        proj = torch.eye(3, **opts)[None] - torch.einsum(
            "ni,nj->nij", normal, normal
        )  # Tangent proj
        gauge_defining_vector = self.gauge_defining_vector(
            data, **opts
        )  # Will define X axis of gauge
        tangent_v = torch.einsum(
            "nij,nj->ni", proj, gauge_defining_vector
        )  # Project vector to tangent plane
        gauge_x = tangent_v / tangent_v.norm(
            p=2, dim=1, keepdim=True
        )  # Normalize and use as X coord
        gauge_y = torch.cross(normal, gauge_x, dim=1)  # Right-handed coord: X x Y = Z <-> Z x X = Y

        idx_to, idx_from = data.edge_index  # (q, p)
        diff = (
            data.pos[idx_from] - data.pos[idx_to]
        )  # We pass message q->p, so express log_p(q) in gauge at p
        r = diff.norm(p=2, dim=1)  # Norm of distance in ambient space
        log_x = (gauge_x[idx_to] * diff).sum(1)
        log_y = (gauge_y[idx_to] * diff).sum(1)
        theta = torch.atan2(
            log_y, log_x
        )  # Project difference in tangent and use polar angle in gauge

        data.edge_coords = torch.stack((r, theta), 1)

        # We parallel transport X coord at q to p and express in gauge at p
        gauge_transported = torch.as_tensor(
            spherical_parallel_transport(
                data.normal[idx_from].numpy(),
                data.normal[idx_to].numpy(),
                gauge_x[idx_from].numpy(),
            ),
            **opts
        )
        connection_x = (gauge_transported * gauge_x[idx_to]).sum(1)
        connection_y = (gauge_transported * gauge_y[idx_to]).sum(1)

        data.connection = torch.atan2(connection_y, connection_x)
        num_neighbours = scatter(torch.ones_like(idx_from), index=idx_to, reduce="sum")
        data.weight = 1.0 / num_neighbours[idx_to]
        data.frame = torch.stack([gauge_x, gauge_y, normal], 1)

        return data
