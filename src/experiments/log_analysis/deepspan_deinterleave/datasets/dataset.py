from collections.abc import Iterable, Iterator, Sized
from typing import Protocol


class Dataset[T](Iterable[T], Sized, Protocol):
    pass


class SizedIterable[T](Dataset[T]):
    def __init__(self, iterable: Iterable[T], length: int):
        self.iterable: Iterable[T] = iterable
        self.length: int = length

    def __iter__(self) -> Iterator[T]:
        return iter(self.iterable)

    def __len__(self) -> int:
        return self.length
