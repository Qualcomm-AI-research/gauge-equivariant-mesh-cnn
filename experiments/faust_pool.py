# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import RunningAverage, Accuracy, Loss
from torch_geometric.datasets import FAUST

from gem_cnn.transform.matrix_features_transform import MatrixFeaturesTransform
from torch_geometric.data import DataLoader
from gem_cnn.transform.scale_mask import ScaleMask
from gem_cnn.transform.multiscale_radius_graph import MultiscaleRadiusGraph

from gem_cnn.nn.gem_res_net_block import GemResNetBlock
from gem_cnn.nn.pool import ParallelTransportPool
from gem_cnn.transform.gem_precomp import GemPrecomp

from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh

max_order = 2

# Number of rings in the radial profile
n_rings = 2

# Ratios used for pooling
ratios = [1, 0.25, 0.25]

# Number of meshes per batch
batch_size = 4

radii = [0.05, 0.14, 0.28]


# 1. Provide a path to load and store the dataset.
# Make sure that you have created a folder 'data' somewhere
# and that you have downloaded and moved the raw datasets there
path = "./data/FAUST"


# 2. Define transformations to be performed on the dataset:
# Transformation that computes a multi-scale radius graph and precomputes the logarithmic map.
pre_transform = T.Compose(
    (
        compute_normals_edges_from_mesh,
        MultiscaleRadiusGraph(ratios, radii, max_neighbours=32),
        SimpleGeometry(),
        MatrixFeaturesTransform(),
    )
)

scale_transforms = [
    T.Compose((ScaleMask(i), GemPrecomp(n_rings, max_order, max_r=radii[i]))) for i in range(3)
]

# Monkey patch to change processed dir, to allow for direct and pool pre-processing
FAUST.processed_dir = osp.join(path, "processed_pool")  # noqa

# 3. Assign and load the datasets.
train_dataset = FAUST(path, train=True, pre_transform=pre_transform)
test_dataset = FAUST(path, train=False, pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
num_nodes = train_dataset[0].num_nodes


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Pre-transform, linear layer

        width = 16

        kwargs = dict(
            n_rings=n_rings,
            band_limit=max_order,
            num_samples=7,
            checkpoint=True,
            batch=100000,
        )

        # Pool
        self.pool1 = ParallelTransportPool(1, unpool=False)
        self.pool2 = ParallelTransportPool(2, unpool=False)
        self.unpool1 = ParallelTransportPool(1, unpool=True)
        self.unpool2 = ParallelTransportPool(2, unpool=True)

        # Stack 1, level 0
        self.conv11 = GemResNetBlock(7, width, 2, max_order, **kwargs)
        self.conv12 = GemResNetBlock(width, width, max_order, max_order, **kwargs)

        # Stack 2, level 1
        self.conv21 = GemResNetBlock(width, width, max_order, max_order, **kwargs)
        self.conv22 = GemResNetBlock(width, width, max_order, max_order, **kwargs)

        # Stack 3, level 2
        self.conv31 = GemResNetBlock(width, width, max_order, max_order, **kwargs)
        self.conv32 = GemResNetBlock(width, width, max_order, max_order, **kwargs)

        # Stack 4, level 1
        self.conv41 = GemResNetBlock(2 * width, width, max_order, max_order, **kwargs)
        self.conv42 = GemResNetBlock(width, width, max_order, max_order, **kwargs)

        # Stack 5, level 0
        self.conv51 = GemResNetBlock(2 * width, width, max_order, max_order, **kwargs)
        self.conv52 = GemResNetBlock(width, width, max_order, 0, **kwargs)

        # Dense final layers
        self.lin1 = nn.Linear(width, 256)
        self.lin2 = nn.Linear(256, num_nodes)

    def forward(self, data):
        data0 = scale_transforms[0](data)
        data1 = scale_transforms[1](data)
        data2 = scale_transforms[2](data)
        attr0 = (data0.edge_index, data0.precomp, data0.connection)
        attr1 = (data1.edge_index, data1.precomp, data1.connection)
        attr2 = (data2.edge_index, data2.precomp, data2.connection)

        x = data.matrix_features

        # Stack 1
        # Select only the edges and precomputed components of the first scale
        x = self.conv11(x, *attr0)
        x = x_l0 = self.conv12(x, *attr0)

        x = self.pool1(x, data)

        x = self.conv21(x, *attr1)
        x = x_l1 = self.conv22(x, *attr1)

        x = self.pool2(x, data)

        x = self.conv31(x, *attr2)
        x = self.conv32(x, *attr2)

        x = self.unpool2(x, data)
        x = torch.cat((x, x_l1), dim=1)

        x = self.conv41(x, *attr1)
        x = self.conv42(x, *attr1)

        x = self.unpool1(x, data)
        x = torch.cat((x, x_l0), dim=1)

        x = self.conv51(x, *attr0)
        x = self.conv52(x, *attr0)

        # Take trivial feature
        x = x[:, :, 0]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


# %%
device = torch.device("cuda")
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
target = (
    torch.arange(num_nodes, dtype=torch.long, device=device).expand(batch_size, num_nodes).flatten()
)

criterion = nn.NLLLoss()


def prepare_batch(batch, device, non_blocking=False):
    data = batch.to(device)
    return data, target.to(device)


trainer = create_supervised_trainer(
    model, optimizer, criterion, device=device, prepare_batch=prepare_batch
)
RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
ProgressBar().attach(trainer, ["loss"])

val_metrics = {"accuracy": Accuracy(), "nll": Loss(criterion)}
evaluator = create_supervised_evaluator(
    model, metrics=val_metrics, device=device, prepare_batch=prepare_batch
)


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    print(
        f"Test Results - Epoch: {trainer.state.epoch} "
        f" Avg accuracy: {metrics['accuracy']:.3f} Avg loss: {metrics['nll']:.2f}"
    )


trainer.run(train_loader, max_epochs=100)
