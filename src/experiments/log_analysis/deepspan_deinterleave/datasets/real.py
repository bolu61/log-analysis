import jax
import jax.numpy as jnp
import pandas as pd


def make_database(seq: pd.Series, window_size, min_length) -> list[jax.Array]:
    def windows():
        for w in seq.rolling(window_size, min_periods=min_length):
            yield jnp.asarray(w.values)

    return list(windows())
