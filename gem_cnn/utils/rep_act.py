# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import torch


def rep_dim(l):
    return 1 if l == 0 else 2


def _rep_act(x, theta):
    """
    :param x: [N, C, 2*order+1]
    :param theta: [N]
    :return: [N, C, 2*order+1]
    """
    y = torch.zeros_like(x)
    order = x.shape[2] // 2
    y[:, :, 0] = x[:, :, 0]
    for l in range(1, order + 1):
        cos = torch.cos(l * theta)[:, None, None]
        sin = torch.sin(l * theta)[:, None, None]
        offset = l * 2 - 1
        y[..., offset : offset + 2 : 2] = (
            cos * x[..., offset : offset + 2 : 2] + -sin * x[..., offset + 1 : offset + 2 : 2]
        )
        y[..., offset + 1 : offset + 2 : 2] = (
            sin * x[..., offset : offset + 2 : 2] + cos * x[..., offset + 1 : offset + 2 : 2]
        )
    return y.view(*y.shape)


class RepAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, theta):
        ctx.args = theta
        return _rep_act(x, theta)

    @staticmethod
    def backward(ctx, grad_y):
        theta = ctx.args
        grad_x = _rep_act(grad_y, -theta)
        return grad_x, None


rep_act = RepAct.apply


def act_so2_vector(th, v):
    """
    :param th: transform angle [N]
    :param v: [N, 2]
    :return: rotate vector by angle
    """
    cos, sin = torch.cos(th), torch.sin(th)
    rotator = torch.stack([cos, -sin, sin, cos], 1).view(-1, 2, 2)
    return torch.einsum("nij,nj->ni", rotator, v)
