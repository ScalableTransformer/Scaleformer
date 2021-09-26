# Scaleformer

A scalable transformer with linear complexity.

Details of implementation and origins of the proposed approach are provided in the [reference paper](./paper/paper-scaleformer.pdf).

## Usage

Recommended way of using this project is in a Docker container built with the provided [Dockerfile](./Dockerfile). For integration with VSCode, extension [vscode-docker](https://github.com/microsoft/vscode-docker) can provide a fully immersive solution. NVidia provides images with [PyTorch](https://pytorch.org/) pre-installed [here](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch). If you do not dispose of a local GPU, an alternative is the use of [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb?utm_source=scs-index).

If you desire to install package in a local environment, simply run `pip install .` or add the flag `-e` for development install.

**NOTE:** to enable GPU in VSCode, add `"runArgs": ["--runtime=nvidia"]` to `.devcontainer/devcontainer.json`.

## Samples

A general workflow sample is provided [here](01-workflow.ipynb). The workflow will produce model in PyTorch `*.pty` format, which later can be used for predictions as per the following snippet. 

```python
import torch

model = torch.load("models/model.pty").to("cpu")
model.predict("Tom is gone")
```