import sys
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
)
from orbax.checkpoint.args import Composite, StandardSave
from tqdm import tqdm

from deepspan.core.deepspan import DeepSpan
from deepspan.train import Trainer
from experiments.deepspan_deinterleave.datasets import make_dataset
from experiments.deepspan_deinterleave.datasets.synthetic import (
    cyclic_sequence_generator,
)

__all__ = ()

NUM_STATES = 8
NUM_CHAINS = 3
NUM_LAYERS = 2
DIM_EMB = 256
DIM_FFN = 1024
LEN_SEQUENCE = 16
LEN_DATASET = 10_000
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
SEED = 0xB0BA

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


def main(*_):
    key = jax.random.key(SEED)
    key_seq, key_train = jax.random.split(key, 2)

    sequences = cyclic_sequence_generator(
        key_seq,
        length=LEN_DATASET * LEN_SEQUENCE,
        num_chains=NUM_CHAINS,
        num_states=NUM_STATES,
    )

    dataset = make_dataset(next(sequences), LEN_SEQUENCE)

    model = DeepSpan(
        num_observations=NUM_STATES * NUM_CHAINS,
        num_layers=NUM_LAYERS,
        dim=DIM_EMB,
        ffn_dim=DIM_FFN,
    )
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batch_size=1,
        dropout_rate=0.2,
        ema_decay=0.9
    )

    checkpoint_directory = Path("checkpoints/deepspan_cyclic").absolute()
    checkpoint_options = CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpoint_manager = CheckpointManager(
        directory=checkpoint_directory,
        options=checkpoint_options,
        item_names=("state", "variables"),
    )

    for epoch in range(NUM_EPOCHS):
        key_train_epoch = jax.random.fold_in(key_train, data=epoch)
        dataset = make_dataset(next(sequences), length=LEN_SEQUENCE)
        s: optax.State
        for l, s, v in (  # type: ignore
            pbar := tqdm(trainer(key_train_epoch), total=len(dataset))
        ):
            pbar.set_description(f"{l:0.8f}")
        checkpoint_manager.save(
            step=epoch,
            args=Composite(state=StandardSave(s), variables=StandardSave(v)),
        )

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(*sys.argv)
