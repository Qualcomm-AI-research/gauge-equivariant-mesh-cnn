# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
from copy import deepcopy

import torch

from gem_cnn.utils.rep_act import act_so2_vector


class GaugeTransformer:
    """
    Transform geometry by a gauge transformation.

    log map v: v -> g_p^{-1} v, so angle th: th - g_p
    transformer g_{q -> p}  ->  g_p^{-1} g_{q->p} g_q
    """

    def __init__(self, transform_angle):
        self.transform_angle = transform_angle

    def __call__(self, data):
        assert len(data.pos) == len(self.transform_angle)
        new_data = deepcopy(data)
        idx_to, idx_from = data.edge_index  # (q, p)
        new_data.connection = (
            -self.transform_angle[idx_to] + self.transform_angle[idx_from] + data.connection
        )
        r, th = data.edge_coords.T
        new_data.edge_coords = torch.stack([r, th - self.transform_angle[idx_to]], 1)
        return new_data
