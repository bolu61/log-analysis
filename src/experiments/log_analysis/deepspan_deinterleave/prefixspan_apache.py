import json
import sys
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

import numpy as np
from prefixspan import make_trie

from deepspan.core.trie import Trie
from deepspan.separate import separate
from experiments.log_analysis.deepspan_deinterleave.datasets.real import (
    Instance,
    make_dataset,
)

SEED = 0xB0BA
LEN_WINDOW = "5ms"
LEN_DATASET_MAX = 2000
LEN_SEQUENCE_MIN = 2
LEN_SEQUENCE_MAX = 16


@dataclass
class Run:
    seed: int
    subject: str
    minsup: int
    len_window: int | str
    len_dataset_max: int = 2000
    len_sequence_min: int = 2
    len_sequence_max: int = 16
    train_split_ratio: float = 0.7


@dataclass
class Grouping:
    instance: Instance
    groups: list[list[int]]


@dataclass
class RunResult:
    run: Run
    groupings: list[Grouping]
    mean_length: float
    median_length: int
    max_length: int


RUNS = [
    Run(SEED, "activemq", 100, "5ms"),
    Run(SEED, "zookeeper", 100, "5ms"),
]


def as_id(event_id: str) -> int:
    return int(event_id[1:])


def get_groups(trie: Trie[int], seq: list[int]) -> list[list[int]]:
    return [
        group
        for group in separate(
            trie=trie,
            seq=seq,
        )
        if len(group) > 0
    ]


def get_event_ids(dataset: list[Instance]) -> list[np.ndarray]:
    return [np.array(i.event_ids, dtype=np.uint64) for i in dataset]


def do_run(run: Run) -> RunResult:
    dataset = make_dataset(
        seed=SEED,
        name=run.subject,
        window_size=run.len_window,
        max_sequence_length=run.len_sequence_max,
        max_dataset_length=run.len_dataset_max,
    )
    split = round(run.train_split_ratio * len(dataset))
    train_dataset, test_dataset = dataset[:split], dataset[split:]

    trie = make_trie(get_event_ids(train_dataset), run.minsup)

    groupings = [
        Grouping(instance=instance, groups=get_groups(trie, instance.event_ids))
        for instance in test_dataset
    ]

    lengths = [len(group) for grouping in groupings for group in grouping.groups]

    return RunResult(
        run=run,
        groupings=groupings,
        mean_length=np.mean(lengths).item(),
        median_length=int(np.median(lengths).item()),
        max_length=np.max(lengths).item(),
    )


def default(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)  # type: ignore
    return str(obj)


def main():
    json.dump({run.subject: do_run(run) for run in RUNS}, sys.stdout, default=default)


if __name__ == "__main__":
    with warnings.catch_warnings(action="ignore"):
        main()
