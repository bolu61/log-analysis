from collections.abc import Sequence

import jax
import jax.numpy as jnp


def grouping_accuracy(xs: Sequence[jax.Array]) -> float:
    a: float = 0
    x: jax.Array
    for x in xs:
        m = jnp.argmax(
            jnp.bincount(
                x,
            )
        )
        a += (x == m).mean().item()
    return a / len(xs)


def grouping_alignment(xs: Sequence[jax.Array]) -> float:
    a: float = 0
    for x in xs:
        a += 1 / jnp.unique(x).size
    return a / len(xs)


def mean_grouping_length(xs: Sequence[jax.Array]) -> float:
    if len(xs) == 0:
        return 0
    return jnp.array([len(x) for x in xs]).mean().item()

def greatest_group_length(xs: Sequence[jax.Array]) -> float:
    if len(xs) == 0:
        return 0
    return jnp.array([len(x) for x in xs]).max().item()

