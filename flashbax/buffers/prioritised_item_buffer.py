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
from flashbax.buffers.item_buffer import validate_item_buffer_args
from flashbax.buffers.prioritised_flat_buffer import validate_priority_exponent
from flashbax.buffers.prioritised_trajectory_buffer import (
    PrioritisedTrajectoryBuffer,
    PrioritisedTrajectoryBufferSample,
    PrioritisedTrajectoryBufferState,
    make_prioritised_trajectory_buffer,
    validate_device,
)
from flashbax.buffers.trajectory_buffer import Experience
from flashbax.utils import add_dim_to_args


def create_prioritised_item_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    add_sequences: bool,
    add_batches: bool,
    priority_exponent: float,
    device: str,
) -> PrioritisedTrajectoryBuffer:
    """Creates a prioritised trajectory buffer that acts as an independent item buffer.

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
        priority_exponent: Priority exponent for sampling. Equivalent to \alpha in the PER paper.
        device: "tpu", "gpu" or "cpu". Depending on chosen device, more optimal functions will be
            used to perform the buffer operations.

    Returns:
        The buffer."""

    validate_item_buffer_args(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
    )

    validate_priority_exponent(priority_exponent)
    if not validate_device(device):
        device = "cpu"

    buffer = make_prioritised_trajectory_buffer(
        max_length_time_axis=max_length,
        min_length_time_axis=min_length,
        add_batch_size=1,
        sample_batch_size=sample_batch_size,
        sample_sequence_length=1,
        period=1,
        priority_exponent=priority_exponent,
        device=device,
    )

    def add_fn(
        state: PrioritisedTrajectoryBufferState, batch: Experience
    ) -> PrioritisedTrajectoryBufferState[Experience]:
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
        state: PrioritisedTrajectoryBufferState, rng_key: PRNGKey
    ) -> PrioritisedTrajectoryBufferSample[Experience]:
        """Samples a batch of items from the buffer."""
        sampled_batch = buffer.sample(state, rng_key)
        priorities = sampled_batch.priorities
        indices = sampled_batch.indices
        sampled_batch = sampled_batch.experience
        sampled_batch = jax.tree_map(lambda x: x.squeeze(axis=1), sampled_batch)
        return PrioritisedTrajectoryBufferSample(
            experience=sampled_batch, indices=indices, priorities=priorities
        )

    return buffer.replace(add=add_fn, sample=sample_fn)  # type: ignore


def make_prioritised_item_buffer(
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    add_sequences: bool = False,
    add_batches: bool = False,
    priority_exponent: float = 0.6,
    device: str = "cpu",
) -> PrioritisedTrajectoryBuffer:
    """Makes a prioritised trajectory buffer act as a independent item buffer.

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
        priority_exponent: Priority exponent for sampling. Equivalent to \alpha in the PER paper.
        device: "tpu", "gpu" or "cpu". Depending on chosen device, more optimal functions will be
            used to perform the buffer operations.

    Returns:
        The buffer."""

    return create_prioritised_item_buffer(
        max_length=max_length,
        min_length=min_length,
        sample_batch_size=sample_batch_size,
        add_sequences=add_sequences,
        add_batches=add_batches,
        priority_exponent=priority_exponent,
        device=device,
    )
