from collections import defaultdict
from collections.abc import Generator, Iterator, Sequence, Iterable

from .database import Database


class DictTrie[K]:
    _children: dict[K, "DictTrie[K]"]
    _count: int

    def __init__(self, count: int):
        self._children = dict()
        self._count = count

    @property
    def count(self):
        return self._count

    def find(self, seq: Sequence[K]) -> "DictTrie[K]":
        def _find(t: DictTrie[K], it: Iterator[K]):
            while True:
                try:
                    k = next(it)
                except StopIteration:
                    return t
                t = t._children[k]

        return _find(self, iter(seq))

    def insert(self, key, t: "DictTrie[K]"):
        self._children[key] = t

    def __str__(self) -> str:
        return (
            "("
            + ",".join((str(k) + str(t) for k, t in self._children.items()))
            + "):"
            + str(self._count)
        )

    def __getitem__(self, key: K) -> "DictTrie[K]":
        return self._children[key]

    def __contains__(self, key: K) -> bool:
        return key in self._children

    def prob(self, key: K) -> float:
        return self._children[key]._count / self._count

    @property
    def keys(self) -> list[K]:
        return sorted(self._children.keys(), reverse=True, key=lambda k: self._children[k]._count)


type Index = tuple[int, int]


def project[
    T
](db: Database[T], indexes: Iterable[Index], s: T) -> Generator[tuple[int, int], None, None]:
    for i, j in indexes:
        try:
            yield i, db[i].index(s, j) + 1
        except ValueError:
            continue
        

def unique[T](it: Iterable[T]) -> Generator[T, None, None]:
    seen = set()
    for t in it:
        if t in seen:
            continue
        yield t
        seen.add(t)


def prefixspan[T](db: Database[T], minsup: int) -> DictTrie[T]:
    def rec(indexes: list[Index]):
        t = DictTrie[T](len(indexes))
        count = defaultdict[T, int](lambda: 0)
        for i, j in indexes:
            for s in unique(db[i][j:]):
                count[s] += 1

        for s, c in count.items():
            if c < minsup:
                continue
            t.insert(s, rec([*project(db, indexes, s)]))

        return t

    return rec([(i, 0) for i in range(len(db))])
