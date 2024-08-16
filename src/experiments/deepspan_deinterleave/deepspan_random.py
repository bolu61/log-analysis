import os
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
from deepspan.train import train
from experiments.deepspan_deinterleave.datasets.batches import make_batches
from experiments.deepspan_deinterleave.datasets import make_dataset
from experiments.deepspan_deinterleave.datasets.synthetic import (
    random_sequence_generator,
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


def make_sequence_generator(key):
    return random_sequence_generator(
        key,
        length=LEN_DATASET * LEN_SEQUENCE,
        num_chains=NUM_CHAINS,
        num_states=NUM_STATES,
    )


def make_model(key):
    model = nn.vmap(
        DeepSpan,
        in_axes=(0, None),
        out_axes=0,
        variable_axes={"params": None},
        split_rngs={"params": False, "dropout": True},
    )(
        num_observations=NUM_STATES * NUM_CHAINS,
        num_layers=NUM_LAYERS,
        dim=DIM_EMB,
        ffn_dim=DIM_FFN,
    )
    variables = model.init(key, jnp.zeros((0, 0), dtype=jnp.int32), 0)
    return model, variables


def make_optimizer(variables):
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    state = optimizer.init(variables)
    return optimizer, state


def enforce_teacher(seqs: list[jax.Array]) -> list[jax.Array]:
    def gen_seqs():
        for seq in seqs:
            for i in range(len(seq)):
                yield seq[:i]

    return [*gen_seqs()]


def main(*_):
    seed = os.getenv("SEED", default=0xB0BAB0BA)
    key = jax.random.key(seed)
    key_seq, key_model, key_train = jax.random.split(key, 3)

    sequences = make_sequence_generator(key_seq)
    model, variables = make_model(key_model)
    optimizer, state = make_optimizer(variables)

    checkpoint_directory = Path("checkpoints/deepspan_random").absolute()
    checkpoint_options = CheckpointManagerOptions(max_to_keep=3, create=True)
    checkpoint_manager = CheckpointManager(
        directory=checkpoint_directory,
        options=checkpoint_options,
        item_names=("state", "variables"),
    )

    for epoch in range(NUM_EPOCHS):
        key_train_epoch = jax.random.fold_in(key_train, data=epoch)
        dataset = make_dataset(next(sequences), length=LEN_SEQUENCE)
        batches = make_batches(dataset, 1)
        for loss, state, variables in (
            pbar := tqdm(
                train(
                    key_train_epoch,
                    optimizer=optimizer,
                    state=state,
                    model=model,
                    variables=variables,
                    dataset=batches,
                    dropout_rate=0.7,
                ),
                total=len(batches),
            )
        ):
            pbar.set_description(f"{loss:0.8f}")
        checkpoint_manager.save(
            step=epoch,
            args=Composite(
                state=StandardSave(state), variables=StandardSave(variables)
            ),
        )

    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(*sys.argv)
