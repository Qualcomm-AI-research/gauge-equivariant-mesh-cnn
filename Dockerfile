FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake && \
    rm -rf /var/lib/apt/lists/*
ADD . /workspace
RUN pip install -f https://data.pyg.org/whl/torch-1.9.0+cu111.html -e .
