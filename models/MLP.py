from flax import linen as nn


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
