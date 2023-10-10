from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
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


class CouplingLayer(nn.Module):
    network_s: nn.Module  # NN to use in the flow for predicting mu and sigma
    network_t: nn.Module  # NN to use in the flow for predicting mu and sigma

    mask: np.ndarray  # Binary mask where 0 denotes that the element should be transformed, and 1 not.
    c_in: int  # Number of input channels

    def setup(self):
        self.scaling_factor = self.param("scaling_factor", nn.initializers.zeros, (self.c_in,))

    def __call__(self, z, reverse=False, orig_img=None):  # ...,ldj, ...
        # Apply network to masked input
        z_in = z * self.mask
        s = self.network_s(z_in)
        t = self.network_t(z_in)

        # Stabilize scaling output
        s_fac = jnp.exp(self.scaling_factor)
        s = nn.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            z = (z + t) * jnp.exp(s)

        else:
            z = (z * jnp.exp(-s)) - t

        return z


class VP_CouplingLayer(nn.Module):
    network_t: nn.Module

    mask: np.ndarray
    c_in: int

    def setup(self):
        pass

    def __call__(self, z, reverse=False, orig_img=None):
        z_in = z * self.mask
        t = self.network_t(z_in)

        t = t * (1 - self.mask)

        if not reverse:
            z = z + t
        else:
            z = z - t

        return z


class SimpleMLP(nn.Module):
    num_hidden: int
    num_layers: int
    num_outputs: int

    def setup(self):
        self.linears = [nn.Dense(features=self.num_hidden) for i in range(self.num_layers)]

    def __call__(self, x):
        for linear in self.linears[:-1]:
            x = linear(x)
            x = nn.tanh(x)

        x = self.linears[-1](x)

        return x


def create_VP_flow(N, num_hidden, d):
    flow_layers = []

    flow_layers += [
        VP_CouplingLayer(
            network_t=SimpleMLP(num_hidden=num_hidden, num_outputs=d),
            mask=generate_mask(i, d),
            c_in=1,
        )
        for i in range(N)
    ]

    flow_model = FlowModel(d, flow_layers)

    return flow_model


def create_flow(N, num_hidden, d):
    flow_layers = []

    flow_layers += [
        CouplingLayer(
            network_s=SimpleMLP(num_hidden=num_hidden, num_outputs=d),
            network_t=SimpleMLP(num_hidden=num_hidden, num_outputs=d),
            mask=generate_mask(i, d),
            c_in=1,
        )
        for i in range(N)
    ]

    flow_model = FlowModel(d, flow_layers)

    return flow_model


def generate_mask(i, d):
    if i % 2 == 0:
        return jax.numpy.array([0.0 for i in range(d)] + [1.0 for i in range(d)])
    else:
        return jax.numpy.array([1.0 for i in range(d)] + [0.0 for i in range(d)])
