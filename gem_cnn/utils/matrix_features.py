# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import torch

from torch_scatter import scatter_sum


def three_matrix_to_so2_irreps(m):
    r"""
    Decompose (rho_1 + rho_0)^{otimes 2} into (3 rho_0 + 2 rho_1 + rho_2)

    :param m: [B, 3, 3]
    :return: ([B, 1], [B, 1], [B, 1], [B, 2], [B, 2], [B, 2])
    """
    v_0_0 = -0.5 * m[:, 0, 1] + 0.5 * m[:, 1, 0]
    v_0_1 = 0.5 * m[:, 0, 0] + 0.5 * m[:, 1, 1]
    v_0_2 = m[:, 2, 2]
    v_1_0 = m[:, 2, :2]
    v_1_1 = m[:, :2, 2]
    v_2_0 = 0.5 * torch.stack([m[:, 0, 1] + m[:, 1, 0], -m[:, 0, 0] + m[:, 1, 1]], 1)
    return v_0_0[:, None], v_0_1[:, None], v_0_2[:, None], v_1_0, v_1_1, v_2_0


def three_sym_matrix_to_so2_features(m):
    """
    Express symmetric (rho_1+rho_0)^{otimes 2} matrix in 2 (rho_0+rho_1+rho_2) SO2 features.

    :param m: [B, 3, 3]
    :return: [B, 2, 5]
    """
    _, v_0_1, v_0_2, v_1_0, _, v_2_0 = three_matrix_to_so2_irreps(m)
    zero = torch.zeros_like(v_1_0)
    return torch.stack(
        [
            torch.cat([v_0_1, v_1_0, v_2_0], 1),
            torch.cat([v_0_2, zero, zero], 1),
        ],
        1,
    )


def three_matrix_to_so2_features(m):
    """
    Express (rho_1+rho_0)^{otimes 2} matrix in 3 (rho_0+rho_1+rho_2) SO2 features.

    :param m: [B, 3, 3]
    :return: [B, 3, 5]
    """
    v_0_0, v_0_1, v_0_2, v_1_0, v_1_1, v_2_0 = three_matrix_to_so2_irreps(m)
    zero = torch.zeros_like(v_1_0)
    return torch.stack(
        [
            torch.cat([v_0_0, v_1_0, v_2_0], 1),
            torch.cat([v_0_1, v_1_1, zero], 1),
            torch.cat([v_0_2, zero, zero], 1),
        ],
        1,
    )


def vector_vector_feature(v_a, v_b, weight, p_idx, frames, symmetric):
    """
    Taking outer product, create matrix feature per pair, average, express in SO2 feature.

    :param v_a: [E, 3]
    :param v_b: [E, 3]
    :param weight: [E]
    :param p_idx: [E] index [0, V)
    :param frames: [V, 3, 3]  per vertex, rows are (X, Y, normal) vectors.
    :param symmetric: bool
    :return: [V, 2/3, 5]  (2 channels if symmetric)
    """
    m_pair = torch.einsum("ni,nj,n->nij", v_a, v_b, weight)
    m_p = scatter_sum(m_pair, p_idx, dim=0) / scatter_sum(weight, p_idx)[:, None, None]
    m_p_gauge = frames @ m_p @ frames.transpose(1, 2)
    return (three_sym_matrix_to_so2_features if symmetric else three_matrix_to_so2_features)(
        m_p_gauge
    )


def matrix_features(edge_index, pos, frames, weight=None):
    """
    Compute feature based on outer product of position difference between neighbouring vertices.

    :param edge_index: [2, M]  (indices of neighbours for M pairs)
    :param weight: [M]  (weight of each pair)
    :param pos: [N, 3]
    :param frames: [N, 3, 3]  for each point, rows are X, Y, normal vectors
    :param max_radius: float
    :return: [N, 7, 5]  7 (rho_0+rho_1+rho_2) features
    """
    p, q = edge_index
    weight = torch.ones(len(p), device=pos.device, dtype=pos.dtype) if weight is None else weight

    d = pos[q] - pos[p]
    normal = frames[q, 2]

    return torch.cat(
        [
            vector_vector_feature(d, d, weight, p, frames, symmetric=True),
            vector_vector_feature(normal, normal, weight, p, frames, symmetric=True),
            vector_vector_feature(d, normal, weight, p, frames, symmetric=False),
        ],
        1,
    )


def so2_feature_to_ambient_vector(v, frames):
    """
    Transform rho_0 + rho_1 feature into ambient 3-vector using frame.

    :param v: [N, C, 3]
    :param frames: [N, 3, 3]  for each point, rows are X, Y, normal vectors
    :return: [N, C, 3]
    """
    return v[:, :, [1, 2, 0]] @ frames  # equiv to 'nci,nix->ncx'


def transform_frames(g, frames):
    c, s, z, o = torch.cos(g), torch.sin(g), torch.zeros_like(g), torch.ones_like(g)
    rho = torch.stack([c, -s, z, s, c, z, z, z, o], 1).view(-1, 3, 3)
    frames_t = rho @ frames
    return frames_t


def test_transformation():
    import math
    from scipy.stats import special_ortho_group
    from gem_cnn.utils.rep_act import rep_act
    import numpy as np

    g = torch.rand(10, dtype=torch.double) * 2 * math.pi
    m = torch.randn(10, 3, 3, dtype=torch.double)
    m_sym = m @ m.transpose(1, 2)  # Make symmetric
    frames = torch.tensor(special_ortho_group(3).rvs(10))
    frames_t = transform_frames(g, frames)
    m_gauge = frames @ m @ frames.transpose(1, 2)
    m_gauge_t = frames_t @ m @ frames_t.transpose(1, 2)
    v = three_matrix_to_so2_features(m_gauge)
    v_t = three_matrix_to_so2_features(m_gauge_t)
    np.testing.assert_allclose(rep_act(v, g), v_t)

    m_sym_gauge = frames @ m_sym @ frames.transpose(1, 2)
    m_sym_gauge_t = frames_t @ m_sym @ frames_t.transpose(1, 2)
    v = three_sym_matrix_to_so2_features(m_sym_gauge)
    v_t = three_sym_matrix_to_so2_features(m_sym_gauge_t)
    np.testing.assert_allclose(rep_act(v, g), v_t)


def test_features():
    import math
    from scipy.stats import special_ortho_group
    from gem_cnn.utils.rep_act import rep_act
    import numpy as np

    num_v = 100
    pos = torch.randn(num_v, 3, dtype=torch.double)
    g = torch.rand(num_v, dtype=torch.double) * 2 * math.pi
    frames = torch.tensor(special_ortho_group(3).rvs(num_v))
    frames_t = transform_frames(g, frames)

    v = matrix_features(pos, frames, 0.5)
    v_t = matrix_features(pos, frames_t, 0.5)
    np.testing.assert_allclose(rep_act(v, g), v_t, atol=1e-14)
