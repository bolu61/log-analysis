import re
import warnings
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import pandas as pd
from dateutil.parser import parse as parse_dt
from logparse.drain3 import parser as drain3
from logparse.parser import Parser

from experiments.log_analysis.deepspan_deinterleave.datasets import Dataset

LOGS_BASE_PATH = Path(__file__).parent / "apache" / "logs"
LOGS_FORMAT_PATTERN = re.compile(
    r"(?P<timestamp>\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d) (\[.*\]\s* - )?\S+\s* \[.*\]\s*( \S+\s* \(.*\)\s*)? - (?P<message>.*)"
)
MIN_FILE_SIZE = 64


def get_paths(name: str) -> Generator[Path, None, None]:
    yield from LOGS_BASE_PATH.glob(f"{name}.*.*-output.txt")


def make_parser(name: str) -> Parser:
    lines: list[str] = []
    for path in get_paths(name):
        with path.open("r") as file:
            if len([*file]) < MIN_FILE_SIZE:
                continue
            lines.extend(file)
    return drain3(lines)


def preprocess(
    parser: Parser,
    lines: Iterable[str],
) -> pd.Series:
    timestamps: list[datetime] = []
    event_ids: list[int] = []
    for line in lines:
        match = LOGS_FORMAT_PATTERN.match(line)
        if match is None:
            continue
        timestamps.append(
            parse_dt(
                match.group("timestamp"),
            )
        )
        event_ids.append(next(parser([match.group("message")])).id)
    return pd.Series(event_ids, timestamps, dtype=int)


def make_database(name: str, window_size, min_length) -> Dataset:
    if not LOGS_BASE_PATH.exists():
        raise RuntimeError(f"{LOGS_BASE_PATH=} does not exist")

    parser = make_parser(name)

    items = []
    for path in get_paths(name):
        with path.open("r") as file:
            seq = preprocess(parser, file)
            if seq.empty:
                warnings.warn(f"no logs parsed in {file=}")
                continue
            seq.sort_index(inplace=True)
        for w in seq.rolling(window_size, min_periods=min_length):
            items.append(jnp.asarray(w.values, copy=False))

    return items
