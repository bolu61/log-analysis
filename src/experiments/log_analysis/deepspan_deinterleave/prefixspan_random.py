import sys
from collections.abc import Generator, Sequence

import jax
import jax.numpy as jnp

from deepspan.core.prefixspan import prefixspan
from deepspan.separate import separate
from experiments.deepspan_deinterleave.datasets.synthetic import (
    random_sequence_generator,
    make_dataset,
)
from experiments.deepspan_deinterleave.metrics.grouping import (
    grouping_accuracy,
    grouping_alignment,
    grouping_length,
)

NUM_STATES = 8
NUM_CHAINS = 3
LEN_SEQUENCE = 24
LEN_DATASET = 10_00
MINSUP = 2_00
SEED = 0xB0BA


def list_sequence(dataset: Sequence[jax.Array]) -> Generator[list[int], None, None]:
    for sample in dataset:
        yield sample.tolist()


def main(*_):
    key = jax.random.key(SEED)

    sequences = random_sequence_generator(
        key,
        length=LEN_DATASET * LEN_SEQUENCE,
        num_chains=NUM_CHAINS,
        num_states=NUM_STATES,
    )

    _, sequence_train = next(sequences)
    dataset = make_dataset(sequence_train, LEN_SEQUENCE)
    trie = prefixspan([*map(lambda a: a.tolist(), dataset)], minsup=MINSUP)

    choices = []
    for group in separate(
        trie=trie,
        seq=jnp.stack(next(sequences)).transpose(),
        maxlen=LEN_SEQUENCE,
        key=lambda cy: cy[1].item(),
    ):
        choices.append(jnp.stack(group)[:, 0])

    print(f"grouping_accuracy: {grouping_accuracy(choices)}")
    print(f"grouping_length: {grouping_length(choices)}")
    print(f"grouping_alignment: {grouping_alignment(choices)}")


if __name__ == "__main__":
    main(*sys.argv)
