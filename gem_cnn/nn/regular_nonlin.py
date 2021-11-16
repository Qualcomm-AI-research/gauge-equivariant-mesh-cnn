# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import numpy as np
import torch
import torch.nn as nn


def create_dft(num_samples, max_n=None, real_adjustment=True):
    assert num_samples % 2 == 1
    max_n = max_n or num_samples // 2
    ns = np.arange(1, max_n + 1)
    x = np.linspace(0, 2 * np.pi, num=num_samples, endpoint=False)
    cosines = np.cos(ns[:, None] * x)
    sines = np.sin(ns[:, None] * x)
    interleaved = np.stack((cosines, sines), 1).reshape(2 * max_n, num_samples)
    if real_adjustment:
        interleaved *= np.sqrt(2)
    dft = np.concatenate((np.ones(num_samples)[None], interleaved)) / np.sqrt(num_samples)

    if real_adjustment and num_samples % 2 == 0:
        dft[-2] *= 1 / np.sqrt(2)

    return dft


class RegularNonlinearity(nn.Module):
    """
    Maps irreps to regular. Then performs non-linearity. Nonlinearity may include batchnorm or dropout.
    """

    def __init__(self, order, num_samples, fn):
        assert num_samples % 2 == 1
        super().__init__()
        self.fn = fn

        self.num_samples = num_samples
        self.order = order
        self.register_buffer(
            "dft", torch.tensor(create_dft(num_samples, order), dtype=torch.float32)
        )

    def forward(self, x):
        """
        :param x: [num_vertices, num_channels, 2 * order + 1]
        :return: [num_vertices, num_channels, 2 * order + 1]
        """
        d = x.shape[2]
        samples = torch.einsum("ncd,dx->ncx", x, self.dft[:d])
        samples_f = self.fn(samples)
        x_f = torch.einsum("ncx,dx->ncd", samples_f, self.dft[:d])
        return x_f

    def __repr__(self):
        return f"{self.__class__.__name__} (order={self.order}, num_samples={self.num_samples}, nonlin={repr(self.fn)})"
