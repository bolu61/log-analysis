from collections.abc import Iterable, Sized
from typing import Protocol
import jax


class Dataset(Iterable[jax.Array], Sized, Protocol):
    pass
