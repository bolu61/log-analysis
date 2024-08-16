import jax
import jax.numpy as jnp
from collections.abc import Sequence


def make_batches(dataset: jax.Array, batch_size: int):
    return dataset.reshape(-1, batch_size, dataset.shape[-1])
