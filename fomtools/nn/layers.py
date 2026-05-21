from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class LinearLayer:
    weight: np.ndarray
    bias: np.ndarray | None = None

    def __post_init__(self):
        self.weight = np.asarray(self.weight, dtype=float)
        if self.bias is not None:
            self.bias = np.asarray(self.bias, dtype=float)


@dataclass
class ReLULayer:
    pass


NNLayer = Union[LinearLayer, ReLULayer]


def from_torch(model) -> list[NNLayer]:
    """
    Convert a PyTorch module to a list of NNLayers.

    Walks ``model.children()`` recursively and converts:
      - ``nn.Linear``  -> ``LinearLayer``
      - ``nn.ReLU``    -> ``ReLULayer``
      - ``nn.Sequential`` / ``nn.Module`` containers -> flattened recursively

    Only ReLU activations and linear layers are supported. Other layer types
    (BatchNorm, dropout, etc.) raise ``ValueError``.

    Parameters
    ----------
    model : torch.nn.Module
        A PyTorch module (typically ``nn.Sequential``).

    Returns
    -------
    list[NNLayer]
    """
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError("PyTorch is required for from_torch(). Install with: pip install torch")

    layers = []

    def _walk(module):
        children = list(module.children())
        if not children:
            if isinstance(module, nn.Linear):
                W = module.weight.detach().cpu().numpy()
                b = module.bias.detach().cpu().numpy() if module.bias is not None else None
                layers.append(LinearLayer(weight=W, bias=b))
            elif isinstance(module, nn.ReLU):
                layers.append(ReLULayer())
            else:
                raise ValueError(
                    f"Unsupported layer type: {type(module).__name__}. "
                    "Only nn.Linear and nn.ReLU are supported."
                )
        else:
            for child in children:
                _walk(child)

    _walk(model)
    return layers


def from_jax(model) -> list[NNLayer]:
    """
    Convert a Flax NNX module to a list of NNLayers.

    Walks the module recursively and converts:
      - ``nnx.Linear``    -> ``LinearLayer``
      - ReLU modules      -> ``ReLULayer``
      - container modules -> flattened recursively

    Only ReLU activations and linear layers are supported. Other layer types
    raise ``ValueError``.

    Parameters
    ----------
    model : flax.nnx.Module
        A Flax NNX module.

    Returns
    -------
    list[NNLayer]
    """
    try:
        from flax import nnx
    except ImportError:
        raise ImportError("Flax is required for from_jax(). Install with: pip install flax")

    layers = []

    def _submodules(module):
        children = []
        for val in vars(module).values():
            if isinstance(val, nnx.Module):
                children.append(val)
            elif isinstance(val, (list, tuple)):
                children.extend(v for v in val if isinstance(v, nnx.Module))
        return children

    def _walk(module):
        children = _submodules(module)
        if not children:
            if isinstance(module, nnx.Linear):
                W = np.asarray(module.kernel.get_value()).T
                bias = getattr(module, "bias", None)
                b = np.asarray(bias.get_value()) if bias is not None else None
                layers.append(LinearLayer(weight=W, bias=b))
            elif "relu" in type(module).__name__.lower():
                layers.append(ReLULayer())
            else:
                raise ValueError(
                    f"Unsupported layer type: {type(module).__name__}. "
                    "Only nnx.Linear and ReLU modules are supported."
                )
        else:
            for child in children:
                _walk(child)

    _walk(model)
    return layers
