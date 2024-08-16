from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Array


@partial(jax.jit, static_argnames=("axis",))
def reglu(x: Array, axis: int = -1) -> Array:
    assert x.shape[axis] % 2 == 0
    a, b = jnp.split(x, 2, axis)
    return a + nn.relu(b)


class FeedForward(nn.Module):
    dim: int
    features: int

    @nn.compact
    def __call__(self, x: Array, dropout_rate=0) -> Array:
        l1 = nn.Dense(self.dim * 2)
        l2 = nn.Dense(self.features)
        dropout = nn.Dropout(rate=dropout_rate, deterministic=False)
        norm = nn.LayerNorm()
        return norm(x + l2(dropout(reglu(l1(x)))))


class GRU(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, xs: Array, dropout_rate=0) -> Array:
        state = self.param("initial_state", nn.initializers.normal(), (self.dim,))
        rnn = nn.RNN(nn.GRUCell(self.dim))
        linear = nn.Dense(self.dim)
        dropout = nn.Dropout(rate=dropout_rate, deterministic=False)
        norm = nn.LayerNorm()
        return norm(xs + linear(dropout(rnn(xs, initial_carry=state))))


class DeepSpanLayer(nn.Module):
    dim: int = 256
    ffn_dim: int = 1024

    @nn.compact
    def __call__(self, xs: Array, dropout_rate=0) -> Array:
        xs = GRU(self.dim)(xs, dropout_rate=dropout_rate)
        xs = FeedForward(self.ffn_dim, self.dim)(xs, dropout_rate=dropout_rate)
        return xs


class DeepSpan(nn.Module):
    num_observations: int = 64
    num_layers: int = 1
    dim: int = 256
    ffn_dim: int = 1024

    @nn.compact
    def __call__(self, xs: Array, dropout_rate=0) -> Array:
        xs = nn.Embed(self.num_observations, self.dim)(xs)
        for _ in range(self.num_layers):
            xs = DeepSpanLayer(self.dim, self.ffn_dim)(xs, dropout_rate=dropout_rate)
        xs = nn.Dense(self.num_observations)(xs)
        return nn.sigmoid(xs)
