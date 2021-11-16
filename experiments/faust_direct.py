# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from torch_geometric.data import DataLoader
from torch_geometric.datasets import FAUST

from gem_cnn.nn.gem_res_net_block import GemResNetBlock
from gem_cnn.transform.gem_precomp import GemPrecomp
from gem_cnn.transform.simple_geometry import SimpleGeometry
from gem_cnn.transform.vector_normals import compute_normals_edges_from_mesh

max_order = 2

# Number of rings in the radial profile
n_rings = 2

# Number of meshes per batch
batch_size = 4


# 1. Provide a path to load and store the dataset.
# Make sure that you have created a folder 'data' somewhere
# and that you have downloaded and moved the raw datasets there
path = "./data/FAUST"


# 2. Define transformations to be performed on the dataset:
# Transformation that computes a multi-scale radius graph and precomputes the logarithmic map.
pre_transform = T.Compose(
    (
        compute_normals_edges_from_mesh,
        SimpleGeometry(),
    )
)

transform = GemPrecomp(n_rings, max_order)

# Monkey patch to change processed dir, to allow for direct and pool pre-processing
FAUST.processed_dir = osp.join(path, "processed_direct")  # noqa

# 3. Assign and load the datasets.
train_dataset = FAUST(path, train=True, pre_transform=pre_transform)
test_dataset = FAUST(path, train=False, pre_transform=pre_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
num_nodes = train_dataset[0].num_nodes


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        width = 16

        kwargs = dict(
            n_rings=n_rings,
            band_limit=max_order,
            num_samples=7,
            checkpoint=True,
            batch=100000,
        )

        self.conv1 = GemResNetBlock(3, width, 0, max_order, **kwargs)
        self.conv2 = GemResNetBlock(width, width, max_order, max_order, **kwargs)
        self.conv3 = GemResNetBlock(width, width, max_order, 0, **kwargs)

        # Dense final layers
        self.lin1 = nn.Linear(width, 256)
        self.lin2 = nn.Linear(256, num_nodes)

    def forward(self, data):
        data0 = transform(data)
        attr0 = (data0.edge_index, data0.precomp, data0.connection)

        x = data.pos[:, :, None]

        x = self.conv1(x, *attr0)
        x = self.conv2(x, *attr0)
        x = self.conv3(x, *attr0)

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
