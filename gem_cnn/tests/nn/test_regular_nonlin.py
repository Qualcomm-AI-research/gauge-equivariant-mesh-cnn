# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import numpy as np
import torch
from math import pi

from gem_cnn.nn.regular_nonlin import RegularNonlinearity
from gem_cnn.utils.rep_act import rep_act


def test_nonlin_gauge_equivariance():
    num_v = 1000
    order, channels = (2, 16)
    dtype = torch.float64

    tests = [(101, 3e-3), (1001, 1e-4), (10001, 1e-5)]

    for num_samples, tolerance in tests:
        transform_angle = torch.rand(num_v, dtype=dtype) * 2 * pi
        nonlin = RegularNonlinearity(order, num_samples=num_samples, fn=torch.nn.ReLU()).to(dtype)
        x = torch.randn(num_v, channels, 2 * order + 1, dtype=dtype)

        x_f = nonlin(x)
        x_f_t = rep_act(x_f, -transform_angle)

        x_t = rep_act(x, -transform_angle)
        x_t_f = nonlin(x_t)

        np.testing.assert_allclose(x_f_t, x_t_f, atol=tolerance)


def test_nonlin_identity():
    num_v = 1000
    order, channels = (2, 16)
    dtype = torch.float64
    tests = [(3, False), (5, True), (11, True)]

    for num_samples, allclose in tests:
        nonlin = RegularNonlinearity(order, num_samples=num_samples, fn=torch.nn.Identity()).to(
            dtype
        )
        x = torch.randn(num_v, channels, 2 * order + 1, dtype=dtype)
        x_f = nonlin(x)
        assert torch.allclose(x, x_f, atol=1e-6) == allclose
