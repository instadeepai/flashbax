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

""""Pure functions defining the prioritised trajectory buffer. The trajectory buffer takes batches
of n-step experience data, where n is the number of time steps within a trajectory. The trajectory
buffer concatenates consecutive batches of experience data along the time axis, retaining their
ordering. This allows for random sampling of the trajectories within the buffer. The prioritised
trajectory buffer associates priorities with each subsequence of experience data broken up by the
period. Prioritisation is implemented as done in the PER paper https://arxiv.org/abs/1511.05952.
"""


import functools
import warnings
from typing import TYPE_CHECKING, Callable, Generic, Optional, Tuple

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax
import jax.numpy as jnp
from jax import Array

from flashbax import utils
from flashbax.buffers import sum_tree, trajectory_buffer
from flashbax.buffers.sum_tree import SumTreeState
from flashbax.buffers.trajectory_buffer import (
    BufferSample,
    BufferState,
    Experience,
    TrajectoryBuffer,
    TrajectoryBufferSample,
    TrajectoryBufferState,
    can_sample,
    get_invalid_indices,
    validate_trajectory_buffer_args,
)

Priorities = Array  # p in the PER paper
Probabilities = Array  # P in the PER paper
Indices = Array

# We change the function that is used for cpu due to the fact that the
# bincount method on cpu scales poorly with buffer size whereas the scan
# method scales poorly with batch size on the TPU.
SET_BATCH_FN = {
    "tpu": sum_tree.set_batch_bincount,
    "gpu": sum_tree.set_batch_bincount,
    "cpu": sum_tree.set_batch_scan,
}


@dataclass(frozen=True)
class PrioritisedTrajectoryBufferState(TrajectoryBufferState, Generic[Experience]):
    """State of the prioritised trajectory replay buffer.

    priority_state: `SumTree`  storing the priorities of the buffer, used for prioritised sampling
        of the indices corresponding to different subsequences.
    """

    priority_state: SumTreeState


@dataclass(frozen=True)
class PrioritisedTrajectoryBufferSample(TrajectoryBufferSample, Generic[Experience]):
    """Container for samples from the prioritised buffer.

    Attributes:
        indices: Indices corresponding to the sampled experience.
        priorities: Unnormalised priorities of the sampled experience. Will be in the form of
            priority**priority_exponent, where `priority_exponent` is denoted as p
            and `priority_exponent` as \alpha in the PER paper.
    """

    indices: Indices
    priorities: Probabilities


def get_sum_tree_capacity(
    max_length_time_axis: int, period: int, add_batch_size: int
) -> int:
    """Get the capacity of the sum tree."""
    return int((max_length_time_axis // period) * add_batch_size)


def prioritised_init(
    experience: Experience,
    add_batch_size: int,
    max_length_time_axis: int,
    period: int,
) -> PrioritisedTrajectoryBufferState[Experience]:
    """
    Initialise the prioritised buffer state.

    Args:
        experience: A single timestep (e.g. (s,a,r)) used for inferring
            the structure of the experience data that will be saved in the buffer state.
        add_batch_size: Batch size of experience added to the buffer's state using the `add`
            function. I.e. the leading batch size of added experience should have size
            `add_batch_size`.
        max_length_time_axis: Maximum length of the buffer along the time axis (second axis of the
            experience data).
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.

    Returns:
        state: Initial state of the replay buffer. All values are empty as no experience has
            been added yet.
    """
    state = trajectory_buffer.init(experience, add_batch_size, max_length_time_axis)

    # The size of the sum tree is the number of possible items
    # based on the period and row length. This is multiplied by the
    # number of rows. If the buffer is not using prioritised sampling
    # then the size is 0.
    sum_tree_size = get_sum_tree_capacity(max_length_time_axis, period, add_batch_size)
    priority_state = sum_tree.init(sum_tree_size)

    return PrioritisedTrajectoryBufferState(priority_state=priority_state, **state)  # type: ignore


def calculate_item_indices_and_priorities(
    state: PrioritisedTrajectoryBufferState,
    sample_sequence_length: int,
    period: int,
    add_sequence_length: int,
    add_batch_size: int,
    max_length_time_axis: int,
) -> Tuple[Array, Array]:
    """
    Calculate the indices and priority values of the items that will be added to
    the prioritised buffer.

    Args:
        state: State of the buffer.
        sample_sequence_length: Length of the sequences that will be sampled from the buffer.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.
        add_sequence_length: Length of the sequences that will be added to the buffer.
        add_batch_size: Batch size of experience added to the buffer's state using the `add`
            function. I.e. the leading batch size of added experience should have size
            `add_batch_size`.
        max_length_time_axis: Maximum length of the buffer along the time axis.

    Returns:
        item_indices: Indices of the items that will be added to the buffer.
        item_values: Priority values of the items that will be added to the buffer.
    """
    # We get the maximum length of the buffer that is divisible by the period, and then
    # subtract 1 to get the maximum index of the buffer. This is so that we can
    # ensure that our index is never above the maximum item index.
    max_divisible_length = max_length_time_axis - (max_length_time_axis % period)
    max_subsequence_data_index = max_divisible_length - 1

    previous_valid_data_index = get_prev_valid_data_idx(
        max_length_time_axis, max_subsequence_data_index, sample_sequence_length, state
    )

    starting_priority_item_index = _get_starting_priority_item_idx(
        max_length_time_axis, period, previous_valid_data_index, state
    )

    ending_data_index = _get_ending_data_idx(
        add_sequence_length, max_length_time_axis, sample_sequence_length, state
    )

    # We ensure that this index is not above the maximum mappable item index of the buffer.
    ending_priority_item_index = (
        jnp.minimum(ending_data_index, max_subsequence_data_index) // period
    ) + 1

    # We get the maximum number of items that can be created based on
    # the number of steps added and the period.
    max_num_items = (add_sequence_length // period) + 1

    # We get the actual number of items that will be created and use for masking.
    actual_num_items = (ending_priority_item_index - starting_priority_item_index) % (
        max_length_time_axis // period
    )

    priority_indices = _get_priority_indices(
        add_batch_size,
        max_length_time_axis,
        max_num_items,
        period,
        starting_priority_item_index,
    )

    # Get a mask indicating which indices are true items and which are padding.
    priority_mask = jnp.where(jnp.arange(max_num_items) >= actual_num_items, 0, 1)

    unnormalised_probability = _get_unnormalised_prob(
        add_batch_size, max_num_items, priority_mask, state
    )

    return priority_indices, unnormalised_probability


def _get_unnormalised_prob(
    add_batch_size: int,
    max_num_items: int,
    priority_mask: Array,
    state: PrioritisedTrajectoryBufferState,
) -> Array:
    """
    Get the unnormalised probability of the items that will be added to the buffer.

    Args:
        add_batch_size: Batch size of experience added to the buffer's state using the `add`
        max_num_items:  Maximum number of items that can be created
        priority_mask: Mask indicating which indices are true items and which are padding.
        state: Buffer state

    Returns:
        Unnormalised probability of the items that will be added to the buffer.
    """
    # We set the priority of the newly created items to the maximum priority however
    # we set masked items to 0.
    unnormalised_probability = (
        jnp.full((max_num_items,), state.priority_state.max_recorded_priority)
        * priority_mask[None]
    )
    unnormalised_probability = jnp.repeat(
        unnormalised_probability, add_batch_size, axis=0
    ).flatten()
    return unnormalised_probability


def _get_priority_indices(
    add_batch_size: int,
    max_length_time_axis: int,
    max_num_items: int,
    period: int,
    starting_priority_item_index: Array,
) -> Array:
    """
    Get the priority indices of the items that will be added to the buffer.
    Args:
        add_batch_size: Batch size of experience added to the buffer's state using the `add`
        max_length_time_axis: Maximum length of the buffer along the time axis.
        max_num_items: Maximum number of items that can be created
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.
        starting_priority_item_index: Starting priority item index

    Returns:
        The latest priority indices of the items that will be added to the buffer.
    """
    # We then get all the indices of the items that will be created.
    priority_indices = (jnp.arange(max_num_items) + starting_priority_item_index) % (
        max_length_time_axis // period
    )
    # We broadcast and add the max_length_time_axis to each index to reference each newly
    # created item in each add_batch row.
    priority_indices = priority_indices + jnp.arange(add_batch_size)[:, None] * (
        max_length_time_axis // period
    )
    # Flatten the indices to a single dimension to use in the priority array.
    priority_indices = priority_indices.flatten()
    return priority_indices


def _get_ending_data_idx(
    add_sequence_length: int,
    max_length_time_axis: int,
    sample_sequence_length: int,
    state: PrioritisedTrajectoryBufferState,
) -> Array:
    """
    Get the ending data index of the items that will be added to the buffer.
    Args:
        add_sequence_length: Length of the sequence that will be added to the buffer.
        max_length_time_axis: Maximum length of the buffer along the time axis.
        sample_sequence_length: Length of the sequence that will be sampled from the buffer.
        state: Buffer state.

    Returns:
        The ending data index of the items that will be added to the buffer.
    """
    # We then get the final index in priority item index based on how many items
    # were created.
    ending_data_index = (
        state.current_index + add_sequence_length - sample_sequence_length
    )
    # If full, we modulo the index by the max length of the buffer to ensure that
    # it wraps around
    ending_data_index = jnp.where(
        state.is_full, ending_data_index % max_length_time_axis, ending_data_index
    )
    return ending_data_index


def _get_starting_priority_item_idx(
    max_length_time_axis: int,
    period: int,
    previous_valid_data_index: Array,
    state: PrioritisedTrajectoryBufferState,
) -> Array:
    """
    Get the starting priority item index of the items that will be added to the buffer.

    Args:
        max_length_time_axis: Maximum length of the buffer along the time axis.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.
        previous_valid_data_index: Previous valid data index.
        state: Buffer state.

    Returns:
        The starting priority item index of the items that will be added to the buffer.
    """
    # We then convert the data index into the item index.
    starting_priority_item_index = (previous_valid_data_index // period) + 1
    # If full, we mod this index by the maximum length of the item buffer (not the data buffer)
    # otherwise we ensure that it is not negative.
    starting_priority_item_index = jnp.where(
        state.is_full,
        starting_priority_item_index % (max_length_time_axis // period),
        jnp.maximum(starting_priority_item_index, 0),
    )
    return starting_priority_item_index


def get_prev_valid_data_idx(
    max_length_time_axis: int,
    max_subsequence_data_index: int,
    sample_sequence_length: int,
    state: PrioritisedTrajectoryBufferState,
) -> Array:
    """
    Get the index of the previous valid timestep in the buffer.

    Args:
        max_length_time_axis: Maximum length of the buffer along the time axis.
        max_subsequence_data_index: Maximum index of the data buffer.
        sample_sequence_length: Length of the sequence that will be sampled from the buffer.
        state: Buffer state.

    Returns:
        The index of the previous valid timestep in the buffer.
    """
    # We get the index of the previous valid timestep.
    previous_valid_data_index = state.current_index - sample_sequence_length
    # If full, we modulo the index by the max length of the buffer to ensure that
    # it wraps around
    previous_valid_data_index = jnp.where(
        state.is_full,
        previous_valid_data_index % max_length_time_axis,
        previous_valid_data_index,
    )
    # We ensure that this index is not above the maximum mappable item index of the buffer.
    previous_valid_data_index = jnp.minimum(
        previous_valid_data_index, max_subsequence_data_index
    )
    return previous_valid_data_index


def prioritised_add(
    state: PrioritisedTrajectoryBufferState[Experience],
    batch: Experience,
    sample_sequence_length: int,
    period: int,
    device: str,
) -> PrioritisedTrajectoryBufferState[Experience]:
    """
    Add a batch of experience to the prioritised buffer state. Assumes that
    this carries on from the episode where the previous added batch of experience
    ended. For example, if we consider a single trajectory within the batch; if
    the last timestep of the previous added trajectory's was at time `t` then the
    first timestep of the current trajectory will be at time `t + 1`.

    Args:
        state: The buffer state.
        batch: A batch of experience. The leading axis of the pytree is the batch dimension.
            This must match `add_batch_size` and the structure of the experience used
            during initialisation of the buffer state. This batch is added along the time axis of
            the buffer state.
        sample_sequence_length: Length of the sequences that will be sampled from the buffer.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.
        device: "tpu", "gpu" or "cpu". Depending on chosen device, more optimal functions will be
            used to perform the buffer operations.

    Returns:
        A new buffer state with the batch of experience added.
    """
    chex.assert_tree_shape_prefix(batch, utils.get_tree_shape_prefix(state.experience))
    chex.assert_trees_all_equal_dtypes(batch, state.experience)

    num_steps = utils.get_tree_shape_prefix(batch, n_axes=2)[1]
    add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
        state.experience, n_axes=2
    )
    chex.assert_axis_dimension_lt(
        jax.tree_util.tree_leaves(batch)[0], 1, max_length_time_axis
    )

    # Calculate index location in the state where we will assign the batch of experience.
    indices = (jnp.arange(num_steps) + state.current_index) % max_length_time_axis

    # Update the priority storage.
    priority_indices, unnormalised_priorities = calculate_item_indices_and_priorities(
        state,
        sample_sequence_length,
        period,
        num_steps,
        add_batch_size,
        max_length_time_axis,
    )

    # Update the sum tree.
    priority_state = SET_BATCH_FN[device](
        state.priority_state,
        priority_indices,
        unnormalised_priorities,
    )

    # Update the buffer state.
    experience = jax.tree_util.tree_map(
        lambda experience_field, batch_field: experience_field.at[:, indices].set(
            batch_field
        ),
        state.experience,
        batch,
    )
    new_index = state.current_index + num_steps
    is_full = state.is_full | (new_index >= max_length_time_axis)
    new_index = new_index % max_length_time_axis
    state = state.replace(  # type: ignore
        experience=experience,
        current_index=new_index,
        is_full=is_full,
        priority_state=priority_state,
    )

    return state


def prioritised_sample(
    state: PrioritisedTrajectoryBufferState[Experience],
    rng_key: chex.PRNGKey,
    batch_size: int,
    sequence_length: int,
    period: int,
) -> PrioritisedTrajectoryBufferSample[Experience]:
    """
    Sample a batch of trajectories from the buffer.

    Args:
        state: The buffer's state.
        rng_key: Random key.
        batch_size: Batch size of sampled experience.
        sequence_length: Length of trajectory to sample.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.

    Returns:
        A batch of experience.
    """
    add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
        state.experience, n_axes=2
    )

    # Sample items from the priority array.
    item_indices = sum_tree.stratified_sample(state.priority_state, batch_size, rng_key)

    trajectory = _get_sample_trajectories(
        item_indices, max_length_time_axis, period, sequence_length, state
    )

    # There is an edge case where experience from the sum-tree has probability 0.
    # To deal with this we overwrite indices with probability zero with
    # the index that is the most probable within the batch of indices. This slightly biases
    # the sampling, however as this is an edge case it is unlikely to have a significant effect.
    priorities = sum_tree.get(state.priority_state, item_indices)
    most_probable_in_batch_index = jnp.argmax(priorities)
    item_indices = jnp.where(
        priorities == 0, item_indices[most_probable_in_batch_index], item_indices
    )
    priorities = jnp.where(
        priorities == 0, priorities[most_probable_in_batch_index], priorities
    )

    # We get the indices of the items that will be invalid when sampling from the buffer state.
    # If the sampled indices are in the invalid indices, then we replace them with the
    # most probable index in the batch. As with above this is unlikely to occur.
    invalid_item_indices = get_invalid_indices(
        state, sequence_length, period, add_batch_size, max_length_time_axis
    )
    invalid_item_indices = invalid_item_indices.flatten()
    item_indices = jnp.where(
        jnp.any(item_indices[:, None] == invalid_item_indices, axis=-1),
        item_indices[most_probable_in_batch_index],
        item_indices,
    )
    priorities = jnp.where(
        jnp.any(item_indices[:, None] == invalid_item_indices, axis=-1),
        priorities[most_probable_in_batch_index],
        priorities,
    )

    return PrioritisedTrajectoryBufferSample(
        experience=trajectory, indices=item_indices, priorities=priorities
    )


def _get_sample_trajectories(
    item_indices: Array,
    max_length_time_axis: int,
    period: int,
    sequence_length: int,
    state: PrioritisedTrajectoryBufferState,
):
    """
    Get the sampled trajectory from the buffer given the sampled indices.

    Args:
        item_indices:  The indices of the sampled items.
        max_length_time_axis:  The maximum length of the time axis of the buffer.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.
        sequence_length: The length of the sampled trajectory.
        state:  The buffer state.

    Returns:
        The sampled trajectory.
    """
    # Convert the item indices to the indices of the data buffer
    flat_indices = item_indices * period

    # Get the batch index and time index of the sampled items.
    batch_indices = flat_indices // max_length_time_axis
    time_index = flat_indices % max_length_time_axis

    # The buffer is circular, so we can loop back to the start (`% max_length_time_axis`)
    # if the time index is greater than the length. We then add the sequence length to get
    # the end index of the sequence.
    time_indices = (
        jnp.arange(sequence_length) + time_index[:, jnp.newaxis]
    ) % max_length_time_axis

    # Slice the experience in the buffer to get a single trajectory of length sequence_length
    trajectory = jax.tree_util.tree_map(
        lambda x: x[batch_indices[:, jnp.newaxis], time_indices], state.experience
    )
    return trajectory


def set_priorities(
    state: PrioritisedTrajectoryBufferState[Experience],
    indices: Indices,
    priorities: Priorities,
    priority_exponent: float,
    device: str,
) -> PrioritisedTrajectoryBufferState[Experience]:
    """
    Set the priorities in the buffer.
    Args:
        state: Current buffer state.
        indices: Locations in the buffer to set the priority.
        priorities: Priority to be set. Commonly this will be abs(td-error).
        priority_exponent: Priority exponent for sampling. Equivalent to \alpha in the PER paper.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last transitions overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.
        device: "tpu", "gpu" or "cpu". Depending on chosen device, more optimal functions will be
            used to perform the buffer operations.


    Returns:
        A buffer state with adjusted priorities.
    """

    unnormalised_probs = jnp.where(
        priorities == 0, jnp.zeros_like(priorities), priorities**priority_exponent
    )
    priority_state = SET_BATCH_FN[device](
        state.priority_state, indices, unnormalised_probs
    )
    return state.replace(priority_state=priority_state)  # type: ignore


@dataclass(frozen=True)
class PrioritisedTrajectoryBuffer(
    TrajectoryBuffer[Experience, BufferState, BufferSample]
):
    """Pure functions defining the prioritised trajectory buffer. This buffer behaves like a
    trajectory buffer, however it also stores the priorities of the data subsequences in a sum tree.

    Attributes:
        init: A pure function which may be used to initialise the buffer state using a single
            timestep (e.g. (s,a,r)).
        add: A pure function for adding a new batch of experience to the buffer state.
        sample: A pure function for sampling a batch of data from the replay buffer, with a leading
            axis of size (`sample_batch_size`, `sample_sequence_length`). Note `sample_batch_size`
            and `sample_sequence_length` may be different to the batch size and sequence length of
            data added to the state using the `add` function.
        can_sample: Whether the buffer can be sampled from, which is determined by if the
            number of trajectories added to the buffer state is greater than or equal to the
            `min_length`.
        set_priorities: A pure function for setting the priorities of the data subsequences in the
            buffer.

    See `make_prioritised_trajectory_buffer` for how this container is instantiated.
    """

    set_priorities: Callable[
        [BufferState, Indices, Priorities],
        BufferState,
    ]


def validate_priority_exponent(priority_exponent: float):
    """
    Validate the priority exponent.
    Args:
        priority_exponent: Priority exponent for sampling. Equivalent to \alpha in the PER paper.

    Returns:
        None
    """
    if priority_exponent < 0 or priority_exponent > 1:
        raise ValueError(
            "Priority exponent must be greater than or"
            f"equal to 0 and less than or equal to 1, got {priority_exponent}"
        )


def validate_device(device: str):
    """
    Checks that the `device` given is a valid jax device.

    Args:
        device: Device to optimise for.

    Returns:
        None
    """
    # Check that the device is valid.
    if device not in ("cpu", "gpu", "tpu"):
        warnings.warn(
            f"Device must be one of 'cpu', 'gpu' or 'tpu', got '{device}'."
            " Defaulting to 'cpu'",
            stacklevel=1,
        )
        return False
    # Check that the device is available.
    backends = []
    for backend in ["cpu", "gpu", "tpu"]:
        try:
            jax.devices(backend)
        except RuntimeError:
            pass
        else:
            backends.append(backend)
    if device not in backends:
        warnings.warn(
            f"You have specified device={device}, however this device is not available."
            " Defaulting to 'cpu'",
            stacklevel=1,
        )
        return False
    return True


def make_prioritised_trajectory_buffer(
    add_batch_size: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    period: int,
    min_length_time_axis: int,
    max_size: Optional[int] = None,
    max_length_time_axis: Optional[int] = None,
    priority_exponent: float = 0.6,
    device: str = "cpu",
) -> PrioritisedTrajectoryBuffer:
    """Makes a prioritised trajectory buffer.

    Args:
        add_batch_size: Batch size of experience added to the buffer. Used to initialise the leading
            axis of the buffer state's experience.
        sample_batch_size: Batch size of experience returned from the `sample` method of the
            buffer.
        sample_sequence_length: Trajectory length of experience of sampled batches. Note that this
            may differ from the trajectory length of experience added to the buffer.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.
        min_length_time_axis: Minimum length of the buffer (along the time axis) before sampling is
            allowed.
        max_size: Optional argument to specify the size of the buffer based on timesteps.
            This sets the maximum number of timesteps that can be stored in the buffer and sets
            the `max_length_time_axis` to be `max_size`//`add_batch_size`. This allows one to
            control exactly how many timesteps are stored in the buffer. Note that this
            overrides the `max_length_time_axis` argument.
        max_length_time_axis: Optional Argument to specify the maximum length of the buffer in terms
            of time steps within the 'time axis'. The second axis (the time axis) of the buffer
            state's experience field will be of size `max_length_time_axis`.
        priority_exponent: Priority exponent for sampling. Equivalent to \alpha in the PER paper.
        device: "tpu", "gpu" or "cpu". Depending on chosen device, more optimal functions will be
            used to perform the buffer operations.


    Returns:
        A trajectory buffer.
    """
    validate_trajectory_buffer_args(
        max_length_time_axis=max_length_time_axis,
        min_length_time_axis=min_length_time_axis,
        add_batch_size=add_batch_size,
        sample_sequence_length=sample_sequence_length,
        period=period,
        max_size=max_size,
    )

    validate_priority_exponent(priority_exponent)
    if not validate_device(device):
        device = "cpu"

    if sample_sequence_length > min_length_time_axis:
        min_length_time_axis = sample_sequence_length

    if max_size is not None:
        max_length_time_axis = max_size // add_batch_size

    init_fn = functools.partial(
        prioritised_init,
        add_batch_size=add_batch_size,
        max_length_time_axis=max_length_time_axis,
        period=period,
    )
    add_fn = functools.partial(
        prioritised_add,
        sample_sequence_length=sample_sequence_length,
        period=period,
        device=device,
    )
    sample_fn = functools.partial(
        prioritised_sample,
        batch_size=sample_batch_size,
        sequence_length=sample_sequence_length,
        period=period,
    )
    can_sample_fn = functools.partial(
        can_sample, min_length_time_axis=min_length_time_axis
    )

    set_priorities_fn = functools.partial(
        set_priorities, priority_exponent=priority_exponent, device=device
    )

    return PrioritisedTrajectoryBuffer(
        init=init_fn,
        add=add_fn,
        sample=sample_fn,
        can_sample=can_sample_fn,
        set_priorities=set_priorities_fn,
    )
