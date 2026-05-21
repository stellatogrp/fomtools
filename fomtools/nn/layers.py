from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Callable

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
RTOL = 1e-5
ATOL = 1e-6


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
        import inspect
        import collections.abc

        children = []
        try:
            module_vars = vars(module)
        except TypeError:
            return children
        for key, val in module_vars.items():
            if key.startswith("_"):
                continue
            if isinstance(val, nnx.Module):
                children.append(val)
            elif isinstance(val, (list, tuple)):
                children.extend(v for v in val if isinstance(v, nnx.Module))
            elif isinstance(val, collections.abc.Callable) and not inspect.ismethod(val):
                children.append(val)
        return children

    def _walk(module):
        # plain callables (not nnx.Module)
        if not isinstance(module, nnx.Module):
            if callable(module) and is_relu(module):
                layers.append(ReLULayer())
            else:
                raise ValueError(
                    f"Unsupported callable: {module!r}. Only nnx.Linear and ReLU are supported."
                )
            return

        # recognize Linear before calling _submodules so its callable attrs
        # don't prevent it from being treated as a leaf.
        if isinstance(module, nnx.Linear):
            W = np.asarray(module.kernel.get_value()).T
            bias = getattr(module, "bias", None)
            b = np.asarray(bias.get_value()) if bias is not None else None
            layers.append(LinearLayer(weight=W, bias=b))
            return

        children = _submodules(module)
        if not children:
            if is_relu(module):
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


def is_relu(fn: Callable, seed: int = 0, check_grad: bool = True) -> bool:
    """
    Return True if `fn` behaves like ReLU on a list of test inputs.

    Parameters
    ----------
        fn : Callable
            The unknown activation, applied either vectorized or element-wise.

        seed: int
            PRNG seed for the random portion of the test inputs.

        check_grad: bool
            If True, also require fn's gradient to match ReLU's
            (0 for x < 0, 1 for x > 0; points near 0 are skipped because the
            derivative is undefined there).

    Returns
    --------
    bool
        True if `fn` behaves like ReLU on a list of test inputs.
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("Flax is required for from_jax(). Install with: pip install flax")

    def _safe_eval(fn, x):
        """Evaluate fn(x), falling back to element-wise application; None on error."""
        try:
            out = fn(x)
            if jnp.asarray(out).shape == x.shape:
                return out
        except Exception:
            pass
        try:
            return jax.vmap(fn)(x)
        except Exception:
            return None

    structured = jnp.array(
        [0.0, -0.0, 1.0, -1.0, 1e-7, -1e-7, 1e3, -1e3, 1e8, -1e8, 3.14159, -2.71828]
    )
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    uniform = jax.random.uniform(k1, (1000,), minval=-100.0, maxval=100.0)
    normal = jax.random.normal(k2, (1000,)) * 10.0
    x = jnp.concatenate([structured, uniform, normal])

    expected = jnp.maximum(0.0, x)

    got = _safe_eval(fn, x)
    if got is None:
        return False

    if got.shape != x.shape:
        return False

    if not bool(jnp.all(jnp.isfinite(got) == jnp.isfinite(expected))):
        return False
    if not bool(jnp.allclose(got, expected, rtol=RTOL, atol=ATOL)):
        return False

    if check_grad:
        nonzero = x[jnp.abs(x) > 10 * ATOL]  # skip the non-differentiable kink
        expected_g = (nonzero > 0).astype(nonzero.dtype)
        grad_fn = jax.vmap(jax.grad(lambda t: jnp.squeeze(fn(t))))
        g = grad_fn(nonzero)
        if not bool(jnp.allclose(g, expected_g, rtol=RTOL, atol=ATOL)):
            return False

    return True
