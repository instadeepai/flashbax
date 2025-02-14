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
from flashbax.buffers.sum_tree import SumTreeState, get_tree_depth
from flashbax.buffers.trajectory_buffer import (
    BufferSample,
    BufferState,
    Experience,
    TrajectoryBuffer,
    TrajectoryBufferSample,
    TrajectoryBufferState,
    can_sample,
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

    sum_tree_state: `SumTree`  storing the priorities of the buffer, used for prioritised sampling
        of the indices corresponding to different subsequences.
    running_index: Array - like current_index, it keeps track of where we are in the buffer
        however it is never modulo'ed. This is required for calculating newly valid and invalid
        items.
    """

    sum_tree_state: SumTreeState
    running_index: Array


@dataclass(frozen=True)
class PrioritisedTrajectoryBufferSample(TrajectoryBufferSample, Generic[Experience]):
    """Container for samples from the prioritised buffer.

    Attributes:
        indices: Indices corresponding to the sampled experience.
        probabilities: probabilities of the sampled experience.
    """

    indices: Indices
    probabilities: Probabilities


def get_max_divisible_length(max_length_time_axis: int, period: int) -> int:
    """
    Compute the maximum length that is divisible by a given period.

    Args:
        max_length_time_axis (int): The original maximum length along the time axis.
        period (int): The period or stride between items.

    Returns:
        int: The maximum length that is divisible by period.
    """
    # Calculate the remainder when dividing max_length_time_axis by period.
    remainder = max_length_time_axis % period
    # Subtract the remainder to yield the largest multiple of period
    # that does not exceed max_length_time_axis.
    return int(max_length_time_axis - remainder)


def get_num_items_in_row(max_length_time_axis: int, period: int) -> int:
    """
    Calculate the number of items that can fit in a row based on the maximum length and the period.

    Args:
        max_length_time_axis (int): The maximum length along the time axis.
        period (int): The period or stride between items.

    Returns:
        int: The number of items that can fit in the given length.
    """
    # Use floor division to count the number of full periods within max_length_time_axis.
    return int(max_length_time_axis // period)


def get_sum_tree_capacity(
    max_length_time_axis: int, period: int, add_batch_size: int
) -> int:
    """
    Calculate the capacity of the sum tree.

    The sum tree capacity is defined as the total number of items in the tree,
    which is the product of the number of items per row and the batch size.

    Args:
        max_length_time_axis (int): The maximum length along the time axis.
        period (int): The period or stride between items.
        add_batch_size (int): The number of batches to add.

    Returns:
        int: The total capacity of the sum tree.
    """
    # Compute the number of items per row (complete periods) and multiply by the batch size.
    return int(get_num_items_in_row(max_length_time_axis, period) * add_batch_size)


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
    # number of rows.
    sum_tree_size = get_sum_tree_capacity(max_length_time_axis, period, add_batch_size)
    sum_tree_state = sum_tree.init(sum_tree_size)

    # Set the running index - Ideally int64 but we put as int32
    running_index = jnp.array(0, dtype=jnp.int32)

    return PrioritisedTrajectoryBufferState(  # type: ignore
        sum_tree_state=sum_tree_state,
        running_index=running_index,
        **state,
    )


def _get_starting_point(
    breaking_point: Array, period: int, max_length_time_axis: int
) -> Array:
    """
    Calculate the starting point for valid items in the buffer given a breaking point.

    Args:
        breaking_point (Array): The current breaking point (e.g., index before writing).
        period (int): The stride between items.
        max_length_time_axis (int): The maximum length along the time axis of the buffer.

    Returns:
        int: The starting point index (as a JAX array).
    """
    breaking_point -= max_length_time_axis
    breaking_point = jnp.maximum(breaking_point, 0)
    starting_point = (breaking_point + period - 1) // period * period
    return starting_point


def _get_valid_buffer_size(starting_point: Array, breaking_point: Array) -> Array:
    """
    Compute the size of the valid portion of the buffer.

    Args:
        starting_point (Array): The starting index of the valid range.
        breaking_point (Array): The breaking point (end index) of the valid range.

    Returns:
        Array: The size of the valid buffer region.
    """
    return breaking_point - starting_point


def _get_num_valid_items_in_buffer(
    starting_point: Array,
    breaking_point: Array,
    sample_sequence_length: int,
    period: int,
) -> Array:
    """
    Calculate the number of valid items in the buffer.

    This is determined by the valid buffer size (from starting_point to breaking_point),
    adjusted by the sequence length and the period.

    Args:
        starting_point (int): The starting index of the valid range.
        breaking_point (int): The breaking point (end index) of the valid range.
        sample_sequence_length (int): The length of the sequence to sample.
        period (int): The interval between items.

    Returns:
        Array: The number of valid items in the buffer.
    """
    valid_buffer_size = _get_valid_buffer_size(starting_point, breaking_point)
    num_valid_items = (valid_buffer_size - sample_sequence_length) // period + 1
    return num_valid_items


def _get_padding_num(
    max_length_time_axis: int, period: int, add_batch_size: int
) -> int:
    """
    Compute the padding number used for static padding in the sum tree.

    The padding number is computed based on the sum tree capacity and its depth.
    It ensures it is out of range.

    Args:
        max_length_time_axis (int): The maximum length along the time axis.
        period (int): The interval between sampled items.
        add_batch_size (int): The batch size for adding experience.

    Returns:
        int: The padding number to be used.
    """
    capacity = get_sum_tree_capacity(max_length_time_axis, period, add_batch_size)
    tree_depth = get_tree_depth(capacity)
    array_size = 2 ** (tree_depth + 1) - 1
    return array_size + 1


def _get_item_indices_and_mask_in_range(
    range_start: Array,
    range_end: Array,
    max_num_items: int,
    period: int,
    max_length_time_axis: int,
) -> Tuple[Array, Array]:
    """
    Compute the item indices and a mask indicating valid positions for a given range.

    Args:
        range_start (int): The starting data index.
        range_end (int): The ending data index.
        max_num_items (int): The maximum number of items possible.
        period (int): The stride between items.
        max_length_time_axis (int): The maximum length of the time axis in the buffer.

    Returns:
        Tuple[Array, Array]: A tuple containing:
            - possible_item_indices: The computed item indices.
            - data_mask: A binary mask indicating valid positions (1 for valid, 0 for invalid).
    """
    # Get the possible 'data' indices in a static shape
    possible_data_indices = range_start + jnp.arange(
        max_num_items * period, step=period
    )
    # Calculate from this the mask of valid data positions
    data_mask = jnp.where(possible_data_indices < range_end, 1, 0)
    # Modulo the data indices to get the correct buffer range
    possible_data_indices = possible_data_indices % max_length_time_axis
    # Divide by period to get the item indices
    possible_item_indices = possible_data_indices // period
    return possible_item_indices, data_mask


def _calculate_new_item_indices(
    index_before_writing: Array,
    add_sequence_length: int,
    period: int,
    max_length_time_axis: int,
    sample_sequence_length: int,
    add_batch_size: int,
) -> Tuple[Array, Array]:
    """
    Returns arrays containing item indices whose subsequences have become newly valid or invalid
    after appending experience. If a previously valid item is still valid
    but its entire subsequence has been overwritten, it is considered a new valid item.

    Args:
        index_before_writing (int): Index before adding new data.
        add_sequence_length (int): Length of the sequence being added.
        period (int): The interval between valid subsequences.
        max_length_time_axis (int): The maximum length along the time axis.
        sample_sequence_length (int): The length of sequences sampled from the buffer.
        add_batch_size (int): The batch size for adding experience.

    Returns:
        Tuple[Array, Array]: A tuple containing:
            - newly_valid_item_indices: Flattened array of new valid item indices.
            - newly_invalid_item_indices: Flattened array of new invalid item indices.
    """

    # Calculate the valid buffer range before adding: (starting_point_1, ending_point_1)
    starting_point_1 = _get_starting_point(
        index_before_writing, period, max_length_time_axis
    )
    breaking_point_1 = index_before_writing
    num_valid_items_1 = _get_num_valid_items_in_buffer(
        starting_point_1, breaking_point_1, sample_sequence_length, period
    )
    ending_point_1 = starting_point_1 + num_valid_items_1 * period

    # Calculate the valid buffer range after adding: (starting_point_2, ending_point_2)
    index_after_writing = index_before_writing + add_sequence_length
    starting_point_2 = _get_starting_point(
        index_after_writing, period, max_length_time_axis
    )
    breaking_point_2 = index_after_writing
    num_valid_items_2 = _get_num_valid_items_in_buffer(
        starting_point_2, breaking_point_2, sample_sequence_length, period
    )
    ending_point_2 = starting_point_2 + num_valid_items_2 * period

    # Calculate the range of newly valid+fully overwritten indices
    range_to_add = (jnp.maximum(starting_point_2, ending_point_1), ending_point_2)
    # Calculate the range of newly invalid indices i.e. broken item subsequences
    range_to_remove = (starting_point_1, jnp.minimum(starting_point_2, ending_point_1))

    # Calculate the maximum number of items possibly created per row
    max_possible_created_items_per_row = (add_sequence_length // period) + 1
    # Get the unmasked item indices for valid and invalid items
    newly_valid_item_indices, valid_mask = _get_item_indices_and_mask_in_range(
        range_to_add[0],
        range_to_add[1],
        max_possible_created_items_per_row,
        period,
        max_length_time_axis,
    )
    newly_invalid_item_indices, invalid_mask = _get_item_indices_and_mask_in_range(
        range_to_remove[0],
        range_to_remove[1],
        max_possible_created_items_per_row,
        period,
        max_length_time_axis,
    )

    # Add the indices for all other rows in the buffer
    # First we create the offset
    offset = jnp.arange(add_batch_size)[:, None] * get_num_items_in_row(
        max_length_time_axis, period
    )
    # then by adding the offset this broadcasts and creates a matrix of indices
    # where each row corresponds to each row in the buffer
    newly_valid_item_indices = newly_valid_item_indices + offset
    # Get the number we will use for padding
    pad_number = _get_padding_num(max_length_time_axis, period, add_batch_size)
    # then we mask each row
    newly_valid_item_indices = jnp.where(
        valid_mask, newly_valid_item_indices, pad_number
    )
    # now we do the same for the invalid indices
    newly_invalid_item_indices = newly_invalid_item_indices + offset
    newly_invalid_item_indices = jnp.where(
        invalid_mask, newly_invalid_item_indices, pad_number
    )

    # Flatten them as sum tree expects a flattened array of item indices.
    newly_valid_item_indices = newly_valid_item_indices.flatten()
    newly_invalid_item_indices = newly_invalid_item_indices.flatten()

    return newly_valid_item_indices, newly_invalid_item_indices


def _calculate_new_item_priorities(
    sum_tree_state: SumTreeState,
    newly_valid_item_indices: Array,
    newly_invalid_item_indices: Array,
    max_length_time_axis: int,
    period: int,
    add_batch_size: int,
) -> Tuple[Priorities, Priorities]:
    """
    Calculate the new priorities for items that have become valid or invalid.

    For newly valid items, assigns the maximum recorded priority from the sum tree.
    For newly invalid items, assigns a priority of zero.

    Args:
        sum_tree_state (SumTreeState): The current state of the sum tree.
        newly_valid_item_indices (Array): Array of indices for newly valid items.
        newly_invalid_item_indices (Array): Array of indices for newly invalid items.
        max_length_time_axis (int): The maximum length along the time axis.
        period (int): The interval between valid subsequences.
        add_batch_size (int): The batch size for adding experience.

    Returns:
        Tuple[Array, Array]: A tuple containing:
            - new_valid_priorities: Priorities for newly valid items.
            - new_invalid_priorities: Priorities for newly invalid items.
    """
    # Get the padding value
    padding_value = _get_padding_num(max_length_time_axis, period, add_batch_size)
    # Calculate the masked valid priorities
    new_valid_priorities = jnp.full_like(
        newly_valid_item_indices, fill_value=sum_tree_state.max_recorded_priority
    )
    vp_mask = newly_valid_item_indices != padding_value
    new_valid_priorities = new_valid_priorities * vp_mask

    # Get invalid priorities
    new_invalid_priorities = jnp.zeros_like(newly_invalid_item_indices)

    return new_valid_priorities, new_invalid_priorities


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

    add_sequence_length = utils.get_tree_shape_prefix(batch, n_axes=2)[1]
    add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
        state.experience, n_axes=2
    )
    chex.assert_axis_dimension_lteq(
        jax.tree_util.tree_leaves(batch)[0], 1, max_length_time_axis
    )

    # Calculate index location in the state where we will assign the batch of experience.
    data_indices = (
        jnp.arange(add_sequence_length) + state.current_index
    ) % max_length_time_axis

    # Update the buffer state.
    new_experience = jax.tree.map(
        lambda exp_field, batch_field: exp_field.at[:, data_indices].set(batch_field),
        state.experience,
        batch,
    )

    # Calculate which items have become valid/fully overwritten and invalid
    valid_items, invalid_items = _calculate_new_item_indices(
        state.running_index,
        add_sequence_length,
        period,
        max_length_time_axis,
        sample_sequence_length,
        add_batch_size,
    )
    valid_priorities, invalid_priorities = _calculate_new_item_priorities(
        state.sum_tree_state,
        valid_items,
        invalid_items,
        max_length_time_axis,
        period,
        add_batch_size,
    )

    # Update the sum tree.
    # IMPORTANT: we have to update the invalid items first and then the valid items
    # this is due to calculating if an item has become invalid - if it is fully overwritten
    # the old item is invalid however the new item is valid thus both appear in the indices
    # this is an intended effect such that we can accurately keep track of such information.
    # First we set the invalid indices to zero
    new_sum_tree_state = SET_BATCH_FN[device](
        state.sum_tree_state,
        invalid_items,
        invalid_priorities,
    )
    # then the valid indices to their respective max_recorded priorities
    new_sum_tree_state = SET_BATCH_FN[device](
        new_sum_tree_state,
        valid_items,
        valid_priorities,
    )

    # Update buffer pointers and flags
    new_current_index = state.current_index + add_sequence_length
    new_running_index = state.running_index + add_sequence_length
    new_is_full = state.is_full | (new_current_index >= max_length_time_axis)
    new_current_index = new_current_index % max_length_time_axis

    return state.replace(  # type: ignore
        experience=new_experience,
        current_index=new_current_index,
        is_full=new_is_full,
        running_index=new_running_index,
        sum_tree_state=new_sum_tree_state,
    )


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
    _, max_length_time_axis = utils.get_tree_shape_prefix(state.experience, n_axes=2)

    # Sample items from the sum tree.
    item_indices = sum_tree.stratified_sample(state.sum_tree_state, batch_size, rng_key)
    probabilities = sum_tree.get_probability(state.sum_tree_state, item_indices)
    # Get the trajectories based on the item indices.
    trajectory = _get_sample_trajectories(
        item_indices, max_length_time_axis, period, sequence_length, state
    )

    return PrioritisedTrajectoryBufferSample(
        experience=trajectory, indices=item_indices, probabilities=probabilities
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
        priorities: Priority to be set. Commonly this will be abs(td-error) + eps.
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
    sum_tree_state = SET_BATCH_FN[device](
        state.sum_tree_state, indices, unnormalised_probs
    )
    return state.replace(sum_tree_state=sum_tree_state)  # type: ignore


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

    assert max_length_time_axis is not None

    # A constraint on the prioritised trajectory buffer
    # is that the max_length_time_axis must be divisible by the period
    prev_max_length_time_axis = max_length_time_axis
    max_length_time_axis = get_max_divisible_length(max_length_time_axis, period)
    if not max_length_time_axis == prev_max_length_time_axis:
        size_difference = (
            prev_max_length_time_axis * add_batch_size
            - max_length_time_axis * add_batch_size
        )
        print(
            f"""Setting max_length_time_axis to {max_length_time_axis} to make divisible by
            period argument. This results in a total reduction in capacity of {size_difference}.""",
        )

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
