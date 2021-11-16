from setuptools import setup, find_packages

setup(
    name="gem_cnn",
    version="1.0.0",
    packages=find_packages(include=["gem_cnn", "gem_cnn.*"]),
    url="",
    license='BSD 3-clause "Clear" License',
    author="Pim de Haan",
    author_email="pim@qti.qualcomm.com",
    description="Code for Gauge Equivariant Mesh Convolution",
    install_requires=[
        "torch",
        "torch_geometric",
        "torch_scatter",
        "torch_cluster",
        "torch_sparse",
        "tqdm",
        "opt_einsum",
        "trimesh",
        "pytorch-ignite",
        "openmesh",
    ],
)
