FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
RUN rm /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake && \
    rm -rf /var/lib/apt/lists/*
RUN conda install pyg openmesh-python -c pyg -c conda-forge
ADD . /workspace
RUN pip install -e .
