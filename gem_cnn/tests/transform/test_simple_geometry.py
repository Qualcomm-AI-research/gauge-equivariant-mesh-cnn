# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import torch

from gem_cnn.transform.simple_geometry import find_first_neighbour


def test_find_first_neighbour():
    edge_index = torch.tensor(
        [
            [2, 0, 1, 0, 1, 1, 2],
            [0, 5, 3, 1, 1, 5, 7],
        ]
    )
    assert find_first_neighbour(edge_index).tolist() == [5, 3, 0]
