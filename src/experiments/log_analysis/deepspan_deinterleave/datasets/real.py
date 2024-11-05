import re
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import dateutil.parser
import numpy as np
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from logparse.drain3 import parser as drain3
from logparse.parser import Parser

DRAIN3_CONFIG = TemplateMinerConfig()

LOGS_BASE_PATH = Path(__file__).parent / "apache" / "logs"
LOGS_FORMAT_PATTERNS = [
    re.compile(
        r"(?P<timestamp>\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d) (\[.*\]\s* - )?\S+\s* \[.*\]\s*( \S+\s* \(.*\)\s*)? - (?P<message>.*)"
    ),
    re.compile(
        r"(?P<timestamp>\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d) \| \w+\s* \| \w+\s* \| (?P<message>.*)"
    ),
    re.compile(
        r"(?P<timestamp>\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d) \[.*\] - \w+\s* \w+\s* - (?P<message>.*)"
    ),
]
DATETIME_PATTERN = re.compile(
    r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),(\d{3})"
)
MIN_FILE_SIZE = 64


@dataclass
class Instance:
    file: Path
    line_numbers: list[int]
    event_ids: list[int]


@dataclass
class LogRecord:
    timestamp: datetime
    lineno: int
    event_id: int


def get_paths(name: str) -> Generator[Path, None, None]:
    yield from (LOGS_BASE_PATH / name).rglob("*-output.txt")


def make_parser(name: str) -> Callable[[str], int]:
    miner = TemplateMiner(config=DRAIN3_CONFIG)
    for path in get_paths(name):
        with path.open("r") as file:
            for line in file:
                miner.add_log_message(line)

    def parser(line: str) -> int:
        return miner.add_log_message(line)["cluster_id"]

    return parser


def match(line: str) -> re.Match | None:
    for pattern in LOGS_FORMAT_PATTERNS:
        match = pattern.match(line)
        if match is not None:
            return match
    return None


def preprocess(
    parse: Callable[[str], int],
    lines: Iterable[str],
) -> list[LogRecord]:
    records: list[LogRecord] = []
    for lineno, line in enumerate(lines):
        if (m := match(line)) is None:
            continue
        records.append(
            LogRecord(
                timestamp=dateutil.parser.parse(m.group("timestamp")),
                lineno=lineno,
                event_id=parse(m.group("message")),
            )
        )
    return records


def read_file(
    parse: Callable[[str], int], path: Path, window_size, max_sequence_length: int
) -> list[Instance]:
    with path.open("r") as f:
        lines = [*f]
        if len(lines) > 100_000:
            return []
        records = preprocess(parse, lines)
    frame = (
        pd.DataFrame(records, columns=["timestamp", "lineno", "event_id"])
        .set_index("timestamp")
        .sort_index()
    )
    results = []
    for window in frame.rolling(window_size):
        if len(window) < 2:
            continue
        window = window[:max_sequence_length]
        results.append(
            Instance(
                file=path,
                line_numbers=window["lineno"].to_list(),
                event_ids=window["event_id"].to_list(),
            )
        )
    return results


def make_dataset(
    seed: int,
    name: str,
    window_size: int | str,
    max_sequence_length: int,
    max_dataset_length: int,
) -> list[Instance]:
    if not LOGS_BASE_PATH.exists():
        msg = f"{LOGS_BASE_PATH=} does not exist"
        raise RuntimeError(msg)

    parse = make_parser(name)

    def gen_windows() -> Iterator[Instance]:
        futures: list[Future[list[Instance]]] = []
        with ThreadPoolExecutor() as executor:
            for path in get_paths(name):
                futures.append(
                    executor.submit(
                        read_file,
                        parse,
                        path,
                        window_size,
                        max_sequence_length,
                    )
                )
        for future in as_completed(futures):
            yield from future.result()

    def sample_windows(windows):
        rng = np.random.default_rng(seed=seed)
        for i in rng.choice(len(windows), size=(max_dataset_length,), replace=False):
            yield windows[i]

    return [*sample_windows([*gen_windows()])]
