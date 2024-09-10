from pathlib import Path
from random import sample

import jax.numpy as jnp
import pandas as pd
from pandas import Series

from deepspan.core.prefixspan import prefixspan
from deepspan.separate import separate
from experiments.deepspan_deinterleave.datasets.real import make_database
from experiments.deepspan_deinterleave.metrics.grouping import grouping_length

NUM_STATES = 8
NUM_CHAINS = 3
LEN_SEQUENCE = 24
LEN_DATASET = 10_000
MINSUP = 2_000


def as_id(event_id: str) -> int:
    return int(event_id[1:])


def main():
    df = pd.read_csv(
        Path(__file__).parent / "datasets" / "samples" / "OpenSSH_2k.log_structured.csv"
    )
    df.index = pd.to_datetime(df.Date + " " + df.Time)  # type: ignore
    df = df.sort_index()
    sequence = df.EventId.apply(lambda i: int(i[1:]))
    database = sample(make_database(sequence, window_size="8s", min_length=2), 2000)

    trie = prefixspan([*map(lambda a: a.tolist(), database)], minsup=int(len(database) * 0.2))

    groups = []
    for group in separate(
        trie, df.to_records(), maxlen=24, key=lambda x: as_id(x.EventId)
    ):
        groups.append(jnp.array([as_id(g.EventId) for g in group]))

    print(f"grouping_length: {grouping_length(groups)}")


if __name__ == "__main__":
    main()
