# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
from math import pi

import torch_geometric.transforms as T

from gem_cnn.tests.utils import random_geometry
from gem_cnn.transform.gauge_transformer import GaugeTransformer
from gem_cnn.utils.rep_act import rep_act

from gem_cnn.nn.gem_conv import GemConv
from gem_cnn.transform.gem_precomp import GemPrecomp

import torch

from gem_cnn.transform.simple_geometry import SimpleGeometry


def test_conv_gauge_equivariance():
    n_rings = 2
    num_v = 10
    max_order = 4
    gem_precomp = GemPrecomp(n_rings=n_rings, max_order=max_order)
    dtype = torch.float64

    transform_angle = torch.rand(num_v, dtype=dtype) * 2 * pi
    transform = T.Compose(
        (
            SimpleGeometry(),
            gem_precomp,
        )
    )

    transform_t = T.Compose(
        (
            SimpleGeometry(),
            GaugeTransformer(transform_angle),
            gem_precomp,
        )
    )

    data_raw = random_geometry(num_v, edge_p=0.6, dtype=dtype)
    data = transform(data_raw)
    data_t = transform_t(data_raw)

    in_order = 2
    out_order = 2
    channels = (16, 16)
    conv = GemConv(*channels, in_order, out_order, n_rings=n_rings).to(dtype)
    conv.bias.data = torch.rand_like(conv.bias)
    x = torch.randn(num_v, channels[0], 2 * in_order + 1, dtype=dtype)

    x_c = conv(x, data.edge_index, data.precomp, data.connection)
    x_c_t = rep_act(x_c, -transform_angle)

    x_t = rep_act(x, -transform_angle)
    x_t_c = conv(x_t, data_t.edge_index, data_t.precomp, data_t.connection)

    torch.testing.assert_allclose(x_c_t, x_t_c)
