import jax
import jax.numpy as jnp
from collections.abc import Sequence


def grouping_accuracy(xs: Sequence[jax.Array]) -> jax.Array:
    a: jax.Array = jnp.array(0, dtype=jnp.float32)
    x: jax.Array
    for x in xs:
        m = jnp.argmax(jnp.bincount(x,))
        a += (x == m).mean()
    return a / len(xs)


def grouping_alignment(xs: Sequence[jax.Array]) -> jax.Array:
    a: jax.Array = jnp.array(0, dtype=jnp.float32)
    for x in xs:
        a += 1 / jnp.unique(x).size
    return a / len(xs)

def grouping_length(xs: Sequence[jax.Array]) -> jax.Array:
    a: jax.Array = jnp.array(0, dtype=jnp.float32)
    for x in xs:
        a += len(x)
    return a / len(xs)
