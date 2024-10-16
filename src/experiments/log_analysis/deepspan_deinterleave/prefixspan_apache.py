import csv
import sys
import warnings
from collections.abc import Iterable

import jax
import jax.numpy as jnp
from prefixspan import make_trie

from deepspan.core.trie import Trie
from deepspan.separate import separate
from experiments.log_analysis.deepspan_deinterleave.datasets.real import make_dataset
from experiments.log_analysis.deepspan_deinterleave.metrics.grouping import (
    greatest_group_length,
    mean_grouping_length,
)

SEED = 0xB0BA
LEN_WINDOW = "5ms"
LEN_DATASET_MAX = 2000
LEN_SEQUENCE_MIN = 2
LEN_SEQUENCE_MAX = 16


SUBJECTS = [
    ("activemq", 100),
    ("zookeeper", 100)
]


def as_id(event_id: str) -> int:
    return int(event_id[1:])


def get_groups(trie: Trie[int], window: Iterable[int]) -> list[jnp.ndarray]:
    groups = [
        jnp.array(group)
        for group in separate(
            trie=trie,
            seq=window,
        )
        if len(group) > 0
    ]
    return [*map(jnp.array, groups)]


def main():
    key = jax.random.key(SEED)
    writer = csv.writer(sys.stdout)
    writer.writerow(("subject", "avg_mean_grouping_length", "avg_greatest_grouping_length"))
    for subject, minsup in SUBJECTS:
        dataset = make_dataset(
            key=key,
            name=subject,
            window_size=LEN_WINDOW,
            max_sequence_length=LEN_SEQUENCE_MAX,
            max_dataset_length=LEN_DATASET_MAX,
        )
        train_dataset, test_dataset = dataset[:1000], dataset[1000:]
        trie = make_trie(train_dataset, minsup)

        groups = [get_groups(trie, w) for w in test_dataset]
        avg_mean_grouping_length = jnp.array([*map(mean_grouping_length, groups)]).mean()
        avg_greatest_grouping_length = jnp.array([*map(greatest_group_length, groups)]).mean()
        writer.writerow((subject, avg_mean_grouping_length, avg_greatest_grouping_length))


if __name__ == "__main__":
    with warnings.catch_warnings(action="ignore"):
        main()
