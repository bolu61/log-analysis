import sys
from collections.abc import Generator, Sequence

import jax
import jax.numpy as jnp
from experiments.log_analysis.deepspan_deinterleave.datasets.synthetic import (
    make_dataset,
    random_sequence_generator,
)
from experiments.log_analysis.deepspan_deinterleave.metrics.grouping import (
    grouping_accuracy,
    grouping_alignment,
    grouping_length,
)
from prefixspan import prefixspan

from deepspan.separate import separate

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
    trie = prefixspan(dataset, minsup=MINSUP)

    choices = [
        jnp.stack(group)[:, 0]
        for group in separate(
            trie=trie,
            seq=jnp.stack(next(sequences)).transpose(),
            maxlen=LEN_SEQUENCE,
            key=lambda cy: cy[1].item(),
        )
    ]

    sys.stdout.write(f"grouping_accuracy: {grouping_accuracy(choices)}\n")
    sys.stdout.write(f"grouping_length: {grouping_length(choices)}\n")
    sys.stdout.write(f"grouping_alignment: {grouping_alignment(choices)}\n")


if __name__ == "__main__":
    main(*sys.argv)
