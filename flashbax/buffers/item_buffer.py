# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
from chex import PRNGKey

from flashbax import utils
from flashbax.buffers.trajectory_buffer import (
    Experience,
    TrajectoryBuffer,
    TrajectoryBufferSample,
    TrajectoryBufferState,
    make_trajectory_buffer,
)
from flashbax.utils import add_dim_to_args


def validate_sample_batch_size(sample_batch_size: int, max_length: int):
    if sample_batch_size > max_length:
        raise ValueError("sample_batch_size must be less than or equal to max_length")


def validate_min_length(min_length: int, max_length: int):
    if min_length > max_length:
        raise ValueError("min_length used cannot be larger than max_length.")


def validate_item_buffer_args(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
):
    """Validates the arguments for the item buffer."""

    validate_sample_batch_size(sample_batch_size, max_length)
    validate_min_length(min_length, max_length)


def create_item_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    add_sequences: bool,
    add_batches: bool,
) -> TrajectoryBuffer:
    """Creates a trajectory buffer that acts as an independent item buffer.

    Args:
        max_length (int): The maximum length of the buffer.
        min_length (int): The minimum length of the buffer.
        sample_batch_size (int): The batch size of the samples.
        add_sequences (Optional[bool], optional): Whether data is being added in sequences
            to the buffer. If False, single items are being added each time add
            is called. Defaults to False.
        add_batches: (Optional[bool], optional): Whether adding data in batches to the buffer.
            If False, single items (or single sequences of items) are being added each time add
            is called. Defaults to False.

    Returns:
        The buffer."""

    validate_item_buffer_args(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
    )

    buffer = make_trajectory_buffer(
        max_length_time_axis=max_length,
        min_length_time_axis=min_length,
        add_batch_size=1,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=1,
        period=1,
    )

    def add_fn(
        state: TrajectoryBufferState, batch: Experience
    ) -> TrajectoryBufferState[Experience]:
        """Flattens a batch to add items along single time axis."""
        batch_size, seq_len = utils.get_tree_shape_prefix(batch, n_axes=2)
        flattened_batch = jax.tree_map(
            lambda x: x.reshape((1, batch_size * seq_len, *x.shape[2:])), batch
        )
        return buffer.add(state, flattened_batch)

    if not add_batches:
        add_fn = add_dim_to_args(
            add_fn, axis=0, starting_arg_index=1, ending_arg_index=2
        )

    if not add_sequences:
        axis = 1 - int(not add_batches)  # 1 if add_batches else 0
        add_fn = add_dim_to_args(
            add_fn, axis=axis, starting_arg_index=1, ending_arg_index=2
        )

    def sample_fn(
        state: TrajectoryBufferState, rng_key: PRNGKey
    ) -> TrajectoryBufferSample[Experience]:
        """Samples a batch of items from the buffer."""
        sampled_batch = buffer.sample(state, rng_key).experience
        sampled_batch = jax.tree_map(lambda x: x.squeeze(axis=1), sampled_batch)
        return TrajectoryBufferSample(experience=sampled_batch)

    return buffer.replace(add=add_fn, sample=sample_fn)  # type: ignore


def make_item_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    add_sequences: bool = False,
    add_batches: bool = False,
) -> TrajectoryBuffer:
    """Makes a trajectory buffer act as a independent item buffer.

    Args:
        max_length (int): The maximum length of the buffer.
        min_length (int): The minimum length of the buffer.
        sample_batch_size (int): The batch size of the samples.
        add_sequences (Optional[bool], optional): Whether data is being added in sequences
            to the buffer. If False, single items are being added each time add
            is called. Defaults to False.
        add_batches: (Optional[bool], optional): Whether adding data in batches to the buffer.
            If False, single transitions or single sequences are being added each time add
            is called. Defaults to False.

    Returns:
        The buffer."""

    return create_item_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_sequences=add_sequences,
        add_batches=add_batches,
    )
