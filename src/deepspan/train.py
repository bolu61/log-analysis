from collections.abc import Generator, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import optax

from deepspan.core.deepspan import DeepSpan


def bceexp(x, y):
    return -jnp.sum(y * x + (1 - y) * jnp.log(-jnp.expm1(x)))


def make_xy(batch: jax.Array) -> tuple[jax.Array, jax.Array]:
    length = batch.shape[-1]
    return batch[..., 0 : length - 1], batch[..., 1:length]


def make_batches(
    dataset: Sequence[jax.Array],
    batch_size: int,
) -> Generator[jax.Array, None, None]:
    for i in range(0, len(dataset), batch_size):
        yield jnp.stack(dataset[i : i + batch_size])


class Trainer:
    def __init__(
        self,
        model: DeepSpan,
        optimizer: optax.GradientTransformation,
        dataset: Sequence[jax.Array],
        dropout_rate: float,
        ema_decay: float = 0.9,
        batch_size: int = 1,
    ):
        self.model: DeepSpan = model
        self.optimizer: optax.GradientTransformation = optimizer
        self.dataset: Sequence[jax.Array] = dataset
        self.dropout_rate = dropout_rate
        self.ema_decay = ema_decay
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __call__(
        self,
        key: jax.Array,
    ):
        @partial(jax.vmap, in_axes=(0, None, 0))
        def apply(key, params, x):
            return self.model.apply(params, x, self.dropout_rate, rngs={"dropout": key})

        @partial(jax.value_and_grad, argnums=1)
        def compute_loss(key, params, x, y):
            keys = jax.random.split(key, self.batch_size)
            y_hat = jax.nn.log_softmax(apply(keys, params, x))
            y_true = jax.nn.one_hot(y, self.model.num_observations)
            return bceexp(y_hat, y_true)

        @jax.jit
        def step(key, state, params, batch):
            x, y = make_xy(batch)
            value, grad = compute_loss(key, params, x, y)
            updates, state = self.optimizer.update(grad, state, params=params)
            params = optax.apply_updates(params, updates)
            return value, state, params

        key_model, key = jax.random.split(key)
        variables = self.model.init(key_model, jnp.zeros((0, 0), dtype=jnp.int32), self.dropout_rate)
        state = self.optimizer.init(variables)
        batches = make_batches(self.dataset, self.batch_size)

        key_init, key = jax.random.split(key)
        loss_ema, state, variables = step(key_init, state, variables, next(batches))
        yield loss_ema, state, variables

        key_train, key = jax.random.split(key)
        for i, batch in enumerate(batches):
            key_step = jax.random.fold_in(key_train, i)
            loss, state, variables = step(key_step, state, variables, batch)
            loss_ema = self.ema_decay * loss_ema + (1 - self.ema_decay) * loss
            yield loss_ema, state, variables
