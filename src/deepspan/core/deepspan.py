from collections.abc import Callable
from functools import partial

import flax.linen as nn
import jax

__all__ = ["DeepSpan"]


def apply_dropout[**A, B](f: Callable[A, B], dropout_rate: float) -> Callable[A, B]:
    def wrapper(*args: A.args, **kwargs: A.kwargs) -> B:
        kwargs.setdefault("dropout_rate", dropout_rate)
        return f(*args, **kwargs)

    return wrapper


class FeedForward(nn.Module):
    dim: int
    features: int

    @nn.compact
    def __call__(self, x: jax.Array, dropout_rate: float = 0) -> jax.Array:
        l1 = nn.Dense(self.dim)
        l2 = nn.Dense(self.features)
        dropout = nn.Dropout(rate=dropout_rate, deterministic=False)
        norm = nn.LayerNorm()
        return norm(x + l2(dropout(nn.relu(l1(x)))))


class GRU(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, xs: jax.Array, dropout_rate: float = 0) -> jax.Array:
        state = self.param("initial_state", nn.initializers.normal(), (self.dim,))
        cell = nn.GRUCell(self.dim)

        @partial(nn.scan, variable_broadcast="params", split_rngs={"params": False}, in_axes=0, out_axes=0)
        def scan(cell, carry, x):
            return cell(carry, x)

        linear = nn.Dense(self.dim)
        dropout = nn.Dropout(rate=dropout_rate, deterministic=False)
        norm = nn.LayerNorm()
        state, ys = scan(cell, state, xs)
        return norm(xs + linear(dropout(nn.relu(ys))))


class DeepSpanLayer(nn.Module):
    dim: int = 256
    ffn_dim: int = 1024

    @nn.compact
    def __call__(self, xs: jax.Array, dropout_rate: float = 0) -> jax.Array:
        gru = apply_dropout(GRU(self.dim), dropout_rate)
        ffn = apply_dropout(FeedForward(self.ffn_dim, self.dim), dropout_rate)
        return ffn(gru(xs))


class DeepSpan(nn.Module):
    num_observations: int = 64
    num_layers: int = 1
    dim: int = 256
    ffn_dim: int = 1024

    @nn.compact
    def __call__(self, xs: jax.Array, dropout_rate: float = 0) -> jax.Array:
        xs = nn.Embed(self.num_observations, self.dim)(xs)
        for _ in range(self.num_layers):
            xs = DeepSpanLayer(self.dim, self.ffn_dim)(xs, dropout_rate=dropout_rate)
        return nn.Dense(self.num_observations)(xs)
