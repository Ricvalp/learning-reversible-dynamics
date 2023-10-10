from typing import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


class FlowModel(nn.Module):
    d: int
    flows: Sequence[nn.Module]

    def setup(self):
        self.R = jax.numpy.array([1.0 for i in range(self.d)] + [-1.0 for i in range(self.d)])

    def __call__(self, x):
        z = x

        for flow in self.flows:
            z = flow(z, reverse=False)

        z = z * self.R

        for flow in reversed(self.flows):
            z = flow(z, reverse=True)

        z = z * self.R

        return z


class HenonLayer(nn.Module):
    V: nn.Module

    def setup(self):
        self.eta = self.param("eta", nn.initializers.zeros, (1, 2))

    def __call__(self, z, reverse=False):
        if not reverse:
            X = jnp.matmul(z, jnp.array([[0.0, 1.0], [0.0, 0.0]]))
            Y = jnp.matmul(z, jnp.array([[0.0, 0.0], [1.0, 0.0]]))
            ETA = jnp.matmul(self.eta, jnp.array([[1.0, 0.0], [0.0, 0.0]]))
            V = jnp.matmul(self.V(Y), jnp.array([[0.0, 0.0], [0.0, 1.0]]))

            return -X + Y + ETA + V

        else:
            X = jnp.matmul(z, jnp.array([[0.0, 1.0], [0.0, 0.0]]))
            Y = jnp.matmul(z, jnp.array([[0.0, 0.0], [1.0, 0.0]]))
            ETA = jnp.matmul(self.eta, jnp.array([[0.0, 0.0], [0.0, 1.0]]))

            Xbar = jnp.matmul(X - ETA, jnp.array([[0.0, 0.0], [1.0, 0.0]]))
            V = jnp.matmul(self.V(Xbar), jnp.array([[0.0, 0.0], [1.0, 0.0]]))

            return X - Y - ETA + V


class SimpleMLP(nn.Module):
    num_hidden: int
    num_layers: int
    num_outputs: int

    def setup(self):
        self.linears = [nn.Dense(features=self.num_hidden) for i in range(self.num_layers - 1)] + [
            nn.Dense(features=self.num_outputs)
        ]

    def __call__(self, x):
        for linear in self.linears[:-1]:
            x = linear(x)
            x = nn.relu(x)

        x = self.linears[-1](x)

        return x


def create_henon_flow(num_layers_flow, num_layers, num_hidden, d):
    flow_layers = []

    flow_layers += [
        HenonLayer(SimpleMLP(num_layers=num_layers, num_hidden=num_hidden, num_outputs=2))
        for _ in range(num_layers_flow)
    ]

    flow_model = FlowModel(d, flow_layers)

    return flow_model


# def create_henon_layer(num_hidden):

#     V = SimpleMLP(num_hidden=num_hidden, num_outputs=2)

#     hanon_layer = HenonLayer(V)

#     return hanon_layer
