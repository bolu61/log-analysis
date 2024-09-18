import re
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Iterable, cast
import warnings

import jax
import jax.numpy as jnp
import pandas as pd
from dateutil.parser import parse as parse_dt
from logparse.drain3 import parser as drain3
from logparse.parser import Parser

LOGS_BASE_PATH = Path(__file__).parent / "apache" / "logs"
LOGS_FORMAT_PATTERN = re.compile(
    r"(?P<timestamp>\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d) (\[.*\]\s* - )?\S+\s* \[.*\]\s*( \S+\s* \(.*\)\s*)? - (?P<message>.*)"
)
MIN_FILE_SIZE = 64


def get_paths(name: str) -> Generator[Path, None, None]:
    yield from LOGS_BASE_PATH.glob(f"{name}.*.*-output.txt")


def make_parser(name: str) -> Parser:
    lines = []
    for path in get_paths(name):
        with path.open("r") as file:
            if len([*file]) < MIN_FILE_SIZE:
                continue
            lines.extend(file)
    return drain3(lines)


def preprocess(
    parser: Parser, lines: Iterable[str]
) -> Generator[tuple[datetime, int], None, None]:
    for line in lines:
        match = LOGS_FORMAT_PATTERN.match(line)
        if match is None:
            continue
        timestamp = parse_dt(
            match.group("timestamp"),
        )
        event_id = next(parser([match.group("message")])).id
        yield timestamp, event_id


def make_database(name: str, window_size, min_length) -> list[jax.Array]:
    if not LOGS_BASE_PATH.exists():
        raise RuntimeError(f"{LOGS_BASE_PATH=} does not exist")

    parser = make_parser(name)

    def windows():
        for path in get_paths(name):
            with path.open("r") as file:
                records = [*preprocess(parser, file)]
                if len(records) == 0:
                    warnings.warn(f"no logs parsed in {file=}")
                    continue
                seq = pd.DataFrame(records).set_index(0)[1].sort_index()
            for w in seq.rolling(window_size, min_periods=min_length):
                yield jnp.asarray(w.values)

    return list(windows())
