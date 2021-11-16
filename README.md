# Geometric Mesh CNN
The code in this repository is an implementation of the Gauge Equivariant Mesh CNN introduced in the paper [Gauge Equivariant Mesh CNNs: Anisotropic convolutions on geometric graphsDownload PDF](https://openreview.net/forum?id=Jnspzp-oIZE) by Pim de Haan, Maurice Weiler, Taco Cohen and Max Welling, presented at ICLR 2021.


We would like to thank Ruben Wiersma as his implementation of [Harmonic Surface Networks](https://github.com/rubenwiersma/hsn) served as an inspiration for some parts of the code. Furthermore, we would like to thank Julian Suk for beta-testing the code.

## Installation & dependencies
Make sure the following dependencies are installed:
* Python (tested on 3.8)
* Pytorch (tested on 1.8)
* Pytorch Geometric (tested on 1.6.3)
* Conda

Then to install, clone this repository and install the `gem_cnn` package by executing in this directory:
```shell
pip install .
```


### Docker
Alternatively, if you have a GPU with CUDA 11.1 and have set up docker, then you can easily run the experiment at `experiments/shapes.py` in the following way:.

To build the image run in this directory:
```shell
docker build . -t gem_cnn_demo
```
Then to run:
```shell
docker run -it --rm --runtime=nvidia gem_cnn_demo python experiments/shapes.py
```

In order to run the FAUST experiments via Docker, we recommend mounting the local `data` folder inside the docker container by running:
```shell
docker run -it --rm --runtime=nvidia -v $(pwd)/data:/workspace/data gem_cnn_demo python experiments/faust_direct.py
```
Then run once, and follow instructions on how to download the dataset.
Then run again to train the FAUST model.

## Usage
The code implements a graph convolution with [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric).

## Example experiments

In the folder `experiments`, the following examples are given:
- `experiments/shapes.py` a simple toy experiment to classify [geometric shapes](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.GeometricShapes). 
- `experiments/faust_direct.py` an implementation of a network similar the network used in our paper on the [FAUST](http://faust.is.tue.mpg.de/) dataset. It does message passing directly over the edges of the mesh and does not use pooling. The used input features are the non-equivariant XYZ coordinates.
- `experiments/faust_pool.py` is an alternative implementation for FAUST. It uses convolution over larger distances than direct neighbours, pooling and the equivariant matrix features.

All example experiments use [Pytorch-Ignite](https://pytorch.org/ignite/index.html), but the GEM-CNN code does not depend on this.

# Reference
If you find our work useful, please cite
```
@inproceedings{dehaan2021,  
  title={Gauge Equivariant Mesh CNNs: Anisotropic convolutions on geometric graphs},  
  author={Pim de Haan and Maurice Weiler and Taco Cohen and Max Welling}
  booktitle={International Conference on Learning Representations},  
  year={2021},  
  url={https://openreview.net/forum?id=Jnspzp-oIZE}  
}
```

# Export
This software may be subject to U.S. and international export, re-export, or transfer (“export”) laws.  Diversion contrary to U.S. and international law is strictly prohibited.
