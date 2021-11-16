# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import networkx as nx
import torch
from torch_geometric.utils import from_networkx


def random_geometry(num_vertices, edge_p=0.3, dtype=torch.float32):
    graph = nx.fast_gnp_random_graph(num_vertices, edge_p)
    data = from_networkx(graph)
    data.pos = torch.randn(num_vertices, 3, dtype=dtype)
    normal = torch.randn(num_vertices, 3, dtype=dtype)
    data.normal = normal / normal.norm(p=2, dim=1, keepdim=True)
    return data
