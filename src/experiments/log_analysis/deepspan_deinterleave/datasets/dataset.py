from collections.abc import (
    Iterable,
    Iterator,
    Sized,
)
from operator import index
from typing import Protocol, SupportsIndex


class Dataset[T](Sized, Iterable[T], Protocol):
    def __getitem__(self, index: SupportsIndex, /) -> T: ...


class IteratorBuffer[T](Dataset[T]):
    def __init__(self, iterator: Iterator[T], length: int, /):
        self._iterator: Iterator[T] = iterator
        self._buffer: list[T] = []
        self._length: int = length

    def __getitem__(self, i: SupportsIndex, /) -> T:
        i = index(i)
        if i >= self._length:
            raise IndexError(i)
        while len(self._buffer) <= i:
            self._buffer.append(next(self._iterator))
        return self._buffer[i]

    def __iter__(self, /) -> Iterator[T]:
        yield from self._buffer
        while len(self._buffer) < self._length:
            yield (item := next(self._iterator))
            self._buffer.append(item)

    def __len__(self, /) -> int:
        return self._length


test: Dataset[int] = IteratorBuffer(iter(range(10)), 10)
