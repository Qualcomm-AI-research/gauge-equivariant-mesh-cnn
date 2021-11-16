# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import torch
import trimesh


def compute_normals_edges_from_mesh(data):
    mesh = trimesh.Trimesh(vertices=data.pos.numpy(), faces=data.face.numpy().T, process=False)
    data.normal = torch.tensor(
        mesh.vertex_normals.copy(), dtype=data.pos.dtype, device=data.pos.device
    )
    data.edge_index = torch.tensor(mesh.edges.T.copy(), dtype=torch.long, device=data.pos.device)
    return data
