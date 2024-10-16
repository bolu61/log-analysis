import re
from collections.abc import Generator, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import dateutil.parser
import jax
import jax.numpy as jnp
import pandas as pd
from logparse.drain3 import parser as drain3
from logparse.parser import Parser

jax.config.update("jax_enable_x64", True)

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


def get_paths(name: str) -> Generator[Path, None, None]:
    yield from (LOGS_BASE_PATH / name).rglob("*-output.txt")


def make_parser(name: str) -> Parser:
    lines: list[str] = []
    for path in get_paths(name):
        with path.open("r") as file:
            if len([*file]) < MIN_FILE_SIZE:
                continue
            lines.extend(file)
    return drain3(lines)


def match(line: str) -> re.Match | None:
    for pattern in LOGS_FORMAT_PATTERNS:
        match = pattern.match(line)
        if match is not None:
            return match
    return None


def preprocess(
    parser: Parser,
    lines: Iterable[str],
) -> pd.Series:
    timestamps: list[datetime] = []
    event_ids: list[int] = []
    for line in lines:
        if (m := match(line)) is None:
            continue
        timestamps.append(
            dateutil.parser.parse(
                m.group("timestamp"),
            )
        )
        event_ids.append(next(parser([m.group("message")])).id)
    return pd.Series(event_ids, timestamps, dtype=jnp.uint64)


def read_file(
    parser: Parser, path: Path, window_size, max_sequence_length: int
) -> list[jnp.ndarray]:
    with path.open("r") as f:
        lines = [*f]
        if len(lines) > 100_000:
            return []
        seq = preprocess(parser, lines)
    seq.sort_index(inplace=True)
    result = []
    for w in seq.rolling(window_size):
        if len(w) < 2:
            continue
        result.append(
            jnp.asarray(w, dtype=jnp.uint64, copy=False)[:max_sequence_length]
        )
    return result


def make_dataset(
    key: jax.typing.ArrayLike,
    name: str,
    window_size: int | str,
    max_sequence_length: int,
    max_dataset_length: int,
) -> list[jnp.ndarray]:
    if not LOGS_BASE_PATH.exists():
        msg = f"{LOGS_BASE_PATH=} does not exist"
        raise RuntimeError(msg)

    parser = make_parser(name)

    def gen_windows() -> Iterator[jnp.ndarray]:
        futures: list[Future[list[jnp.ndarray]]] = []
        with ThreadPoolExecutor() as executor:
            for path in get_paths(name):
                futures.append(
                    executor.submit(
                        read_file,
                        parser,
                        path,
                        window_size,
                        max_sequence_length,
                    )
                )
        for future in as_completed(futures):
            yield from future.result()

    def sample_windows(windows):
        for i in jax.random.choice(
            key, len(windows), shape=(max_dataset_length,), replace=False
        ):
            yield windows[i]

    return [*sample_windows([*gen_windows()])]
