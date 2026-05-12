
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from fomtools.nn import from_torch

class ReLUFFNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def extract_layers(self):
        result = []
        for module in self.net:
            if isinstance(module, nn.Linear):
                result.append({
                    "W"    : module.weight.detach(),
                    "b"    : module.bias.detach(),
                    "relu" : False,
                })
            elif isinstance(module, nn.ReLU):
                result[-1]["relu"] = True
        return result

def get_model(model=None):
    if model is None:
        input_dim   = 4
        hidden_dims = [32, 32]
        output_dim  = 2

        model = ReLUFFNet(input_dim, hidden_dims, output_dim)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn   = nn.MSELoss()

        # train the model
        for step in range(500):
            x    = torch.randn(64, input_dim)
            y    = torch.randn(64, output_dim)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


@pytest.fixture(scope="module")
def trained_model():
    torch.manual_seed(42)
    return get_model()


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(42)
    return ReLUFFNet(input_dim=2, hidden_dims=[4], output_dim=2)


def test_nn_solve(vp, small_model):
    layers = from_torch(small_model)

    input_expr = vp.add_param(2, lb=-1.0, ub=1.0)
    # todo: combine it with other steps
    output_expr = vp.neural_network(input_expr, layers)

    vp.set_objective_norm(output_expr, np.inf)
    result = vp.solve()

    assert result is not None
    assert result >= 0
