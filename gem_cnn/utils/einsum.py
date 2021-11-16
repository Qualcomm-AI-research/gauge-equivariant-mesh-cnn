# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
from functools import lru_cache

from opt_einsum import contract_expression


@lru_cache(1000)
def get_contract_expression(eqn, s):
    return contract_expression(eqn, *s, optimize="optimal")


def einsum(eqn, *ops):
    """
    Wrapper around opt_einsum, which wraps around torch.einsum.
    It caches the contraction order for
    :param eqn:
    :param ops:
    :return:
    """
    s = tuple(tuple(t.shape) for t in ops)
    expr = get_contract_expression(eqn, s)
    res = expr(*ops, backend="torch")
    return res
