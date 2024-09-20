from collections.abc import Generator
from itertools import count

import jax
from jaxtyping import Array

from deepspan.core.hmm import interleaved_cyclic_hmm, interleaved_random_hmm


def cyclic_sequence_generator(
    key: jax.Array, length: int, num_chains: int, num_states: int
) -> Generator[tuple[jax.Array, jax.Array], None, None]:
    mc = interleaved_cyclic_hmm(num_chains=num_chains, num_states=num_states)
    key_init, key_state, key_iter = jax.random.split(key, 3)
    variables = mc.init(key_init, jax.random.key(0), jax.numpy.array([0]))

    state = mc.apply(variables, key_state, method=mc.sample)

    @jax.jit
    def gen_next(state: jax.Array, key: jax.Array):
        (s, c), y = mc.apply(variables, key, state)
        return s, (c, y)

    for c in count(start=1):
        keys = jax.random.fold_in(key_iter, c)
        state, (cs, ys) = jax.lax.scan(gen_next, state, jax.random.split(keys, length))
        yield cs, ys


def random_sequence_generator(
    key: jax.Array, length: int, num_chains: int, num_states: int
) -> Generator[tuple[jax.Array, jax.Array], None, None]:
    mc = interleaved_random_hmm(num_chains=num_chains, num_states=num_states)
    variables = mc.init(jax.random.fold_in(key, 0), jax.random.key(0), jax.numpy.array([0]))

    state: jax.Array = mc.apply(variables, jax.random.fold_in(key, 0), method=mc.sample)  # type: ignore

    @jax.jit
    def gen_next(state: jax.Array, key: jax.Array):
        (s, c), y = mc.apply(variables, key, state)
        return s, (c, y)

    for c in count(start=1):
        keys = jax.random.fold_in(key, c)
        state, (cs, ys) = jax.lax.scan(gen_next, state, jax.random.split(keys, length))
        yield cs, ys


def make_dataset(seq: Array, length: int) -> list[Array]:
    return [seq[i : i + length] for i in range(0, len(seq), length)]
