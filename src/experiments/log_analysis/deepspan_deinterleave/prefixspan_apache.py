from pathlib import Path
from random import sample

import jax.numpy as jnp
import pandas as pd
from pandas import Series

from deepspan.core.prefixspan import prefixspan
from deepspan.separate import separate
from experiments.log_analysis.deepspan_deinterleave.datasets.real import make_database
from experiments.log_analysis.deepspan_deinterleave.metrics.grouping import grouping_length

NUM_STATES = 8
NUM_CHAINS = 3
LEN_SEQUENCE = 24
LEN_DATASET = 10_000
MINSUP = 2_000


SUBJECTS = ["org.apache.zookeeper"]


def as_id(event_id: str) -> int:
    return int(event_id[1:])


def main():
    for subject in SUBJECTS:
        database = make_database(subject, window_size="8s", min_length=2)
        print(database)
        return
    return

    trie = prefixspan([*map(lambda a: a.tolist(), database)], minsup=int(len(database) * 0.2))

    groups = []
    for group in separate(
        trie, df.to_records(), maxlen=24, key=lambda x: as_id(x.EventId)
    ):
        groups.append(jnp.array([as_id(g.EventId) for g in group]))

    print(f"grouping_length: {grouping_length(groups)}")


if __name__ == "__main__":
    main()
