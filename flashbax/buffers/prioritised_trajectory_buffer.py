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

# The terminology in this code can be confusing
# The main points to understand is the concept of 'items' vs 'data'
# Items refer to each individual subsequence that can be sampled where as
# data refers to the actual data stored in the buffer.


@dataclass(frozen=True)
class PrioritisedTrajectoryBufferState(TrajectoryBufferState, Generic[Experience]):
    """State of the prioritised trajectory replay buffer.

    sum_tree_state: `SumTree`  storing the priorities of the buffer, used for prioritised sampling
        of the indices corresponding to different subsequences.
    insert_count: the number of times a data point has been inserted into a cell.
    """

    sum_tree_state: SumTreeState
    insert_count: Array


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


def get_num_items_per_row(max_length_time_axis: int, period: int) -> int:
    """This calculates the number of 'items' in a row of the data buffer."""
    return int(max_length_time_axis // period)


def get_total_num_items(
    max_length_time_axis: int, period: int, add_batch_size: int
) -> int:
    """This calculates the number of possible unique subsequences
    that can be sampled in the entire buffer."""
    return int(get_num_items_per_row(max_length_time_axis, period) * add_batch_size)


def get_sum_tree_capacity(
    max_length_time_axis: int, period: int, add_batch_size: int
) -> int:
    """Get the capacity of the sum tree."""
    return get_total_num_items(max_length_time_axis, period, add_batch_size)


def item_index_to_batch_and_time_index(
    item_indices: Array, period: int, max_length_time_axis: int
) -> Tuple[Array, Array]:
    """This converts from 'item' index which has a range of 0
    to 'get_num_items' into the batch and time index of the actual data storage."""

    # Convert the item indices to the indices of the data buffer
    flat_indices = item_indices * period
    batch_indices = flat_indices // max_length_time_axis
    time_indices = flat_indices % max_length_time_axis

    return batch_indices, time_indices


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

    # Initialise the insert count.
    # This is a useful measure to check for overwritten subsequences.
    insert_count = jnp.zeros((add_batch_size, max_length_time_axis))

    return PrioritisedTrajectoryBufferState(
        sum_tree_state=sum_tree_state, insert_count=insert_count, **state
    )  # type: ignore


def _get_prev_valid_data_idx(
    max_length_time_axis: int,
    max_subsequence_data_index: int,
    sample_sequence_length: int,
    state: PrioritisedTrajectoryBufferState,
) -> Array:
    """
    Get the index of the previous valid data cell in the buffer.

    Args:
        max_length_time_axis: Maximum length of the buffer along the time axis.
        max_subsequence_data_index: Maximum index of the data buffer.
        sample_sequence_length: Length of the sequence that will be sampled from the buffer.
        state: Buffer state.

    Returns:
        The index of the previous valid data cell in the buffer.
    """
    # We get the index of the previous valid timestep.
    # By valid, it means we could sample from this index and return a
    # contiguous trajectory that has not been broken partially.
    previous_valid_data_index = state.current_index - sample_sequence_length
    # If full, we modulo the index by the max length of the buffer to ensure that
    # it wraps around. If we do not have a full buffer, we ensure that it is
    # not below -1 where -1 essentially means that we have no valid data index
    previous_valid_data_index = jnp.where(
        state.is_full,
        previous_valid_data_index % max_length_time_axis,
        jnp.maximum(previous_valid_data_index, -1),
    )
    # We ensure that this index is not above the maximum mappable item index of the buffer.
    # The reason is because there is no 'item' that is above the period divisible max length axis.
    previous_valid_data_index = jnp.minimum(
        previous_valid_data_index, max_subsequence_data_index
    )
    return previous_valid_data_index


def _get_starting_item_idx(
    max_length_time_axis: int,
    period: int,
    previous_valid_data_index: Array,
    state: PrioritisedTrajectoryBufferState,
) -> Array:
    """
    Get the starting 'item' index of the items that will be added to the buffer.

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
    # We then convert the data index into its corresponding 'item' index.
    # and add 1 since we are referring to the priority buffer item where the new data will be
    # added.
    starting_item_index = (previous_valid_data_index // period) + 1
    # If full, we mod this index by the maximum length of the 'item' buffer (not the data buffer)
    # otherwise we ensure that it is not negative. The latter ensures that we
    # always start with a valid item index.
    starting_item_index = jnp.where(
        state.is_full,
        starting_item_index % get_num_items_per_row(max_length_time_axis, period),
        jnp.maximum(starting_item_index, 0),
    )
    return starting_item_index


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
    # We then get the final index in the item buffer based on how many items
    # were created.
    ending_data_index = (
        state.current_index + add_sequence_length - sample_sequence_length
    )
    # If full, we modulo the index by the max length of the buffer to ensure that
    # it wraps around. Otherwise, if the buffer is not full we ensure the ending data index is non
    # negative
    ending_data_index = jnp.where(
        state.is_full,
        ending_data_index % max_length_time_axis,
        jnp.maximum(ending_data_index, 0),
    )
    return ending_data_index


def _get_item_indices_and_priorities(
    state: PrioritisedTrajectoryBufferState,
    sample_sequence_length: int,
    period: int,
    add_sequence_length: int,
    add_batch_size: int,
    max_length_time_axis: int,
) -> Tuple[Array, Array]:
    """Calculate the indices and priority values of the items that will be added to
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
        item_values: Priority values of the items that will be added to the buffer."""

    # Due to supporting any period, it is possible that there are strange division issues
    # Thus we find what the largest max_length_axis that is divisible by the period.
    max_divisible_by_period_length = max_length_time_axis - (
        max_length_time_axis % period
    )
    # Knowing this, we simply minus 1 to get the index of the last possible cell.
    max_subsequence_data_index = max_divisible_by_period_length - 1

    # Get the previous data index value that is valid to start sampling from.
    previous_valid_data_index = _get_prev_valid_data_idx(
        max_length_time_axis, max_subsequence_data_index, sample_sequence_length, state
    )
    # Get the starting index of the 'item' buffer/array that we will start writing from
    starting_item_index = _get_starting_item_idx(
        max_length_time_axis, period, previous_valid_data_index, state
    )

    # Get the ending data index
    ending_data_index = _get_ending_data_idx(
        add_sequence_length, max_length_time_axis, sample_sequence_length, state
    )
    # We ensure that this index is not above the maximum mappable data index of the buffer.
    ending_data_index = jnp.minimum(ending_data_index, max_subsequence_data_index)

    # We now convert the data index to an item index
    ending_item_index = (ending_data_index // period) + 1

    # We get the maximum number of items that can be created based on
    # the number of steps added and the period.
    max_num_items = (add_sequence_length // period) + 1

    # We get the actual number of items that will be created and use for masking.
    actual_num_items_given_full = (
        ending_item_index - starting_item_index
    ) % get_num_items_per_row(max_length_time_axis, period)
    # If not full, we simply take the maximum
    actual_num_items_given_not_full = jnp.maximum(
        0, (ending_item_index - starting_item_index)
    )

    actual_num_items = jax.lax.select(
        state.is_full, actual_num_items_given_full, actual_num_items_given_not_full
    )

    # We then get all the indices of the items that will be created.
    item_indices = (
        jnp.arange(max_num_items) + starting_item_index
    ) % get_num_items_per_row(max_length_time_axis, period)
    # We add the row offset to get the item indices for all rows not just the first one.
    item_indices = item_indices + jnp.arange(add_batch_size)[
        :, None
    ] * get_num_items_per_row(max_length_time_axis, period)
    # Flatten the indices to a single dimension to use in the item array.
    item_indices = item_indices.flatten()
    # Create the valid mask
    valid_mask = jnp.where(jnp.arange(max_num_items) >= actual_num_items, 0, 1)

    # We set the priority of the newly created items to the maximum priority however
    # we set masked items to 0.
    new_priorities = jnp.full(
        (max_num_items,), fill_value=state.sum_tree_state.max_recorded_priority
    )
    new_priorities = new_priorities * valid_mask[None]

    # We repeat the new priorities for each row
    new_priorities = jnp.repeat(new_priorities, add_batch_size, axis=0).flatten()

    return item_indices, new_priorities


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
    time_data_indices = (
        jnp.arange(add_sequence_length) + state.current_index
    ) % max_length_time_axis

    # Update the buffer data.
    experience = jax.tree_util.tree_map(
        lambda experience_field, batch_field: experience_field.at[
            :, time_data_indices
        ].set(batch_field),
        state.experience,
        batch,
    )
    # Update the insert count
    new_insert_count = jax.tree.map(
        lambda insert_count_field: insert_count_field.at[:, time_data_indices].add(1),
        state.insert_count,
    )

    # Now we have to calculate the item indices and priorities of the newly added data.
    item_indices, item_priorities = _get_item_indices_and_priorities(
        state,
        sample_sequence_length,
        period,
        add_sequence_length,
        add_batch_size,
        max_length_time_axis,
    )

    # Update the sum tree.
    sum_tree_state = SET_BATCH_FN[device](
        state.sum_tree_state,
        item_indices,
        item_priorities,
    )

    # Update the other state metrics
    new_index = state.current_index + add_sequence_length
    is_full = state.is_full | (new_index >= max_length_time_axis)
    new_index = new_index % max_length_time_axis

    state = state.replace(  # type: ignore
        experience=experience,
        current_index=new_index,
        is_full=is_full,
        sum_tree_state=sum_tree_state,
        insert_count=new_insert_count,
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
    _, max_length_time_axis = utils.get_tree_shape_prefix(state.experience, n_axes=2)

    # Sample from the sum tree
    sampled_item_indices = sum_tree.stratified_sample(
        state.sum_tree_state, batch_size, rng_key
    )

    # Get the priorities at these indices
    priorities = sum_tree.get(state.sum_tree_state, sampled_item_indices)

    # Convert flat (item) indices -> row/col indices for buffer
    batch_indices, time_indices = item_index_to_batch_and_time_index(
        sampled_item_indices, period, max_length_time_axis
    )

    # Check validity: if any cell in [start_t..start_t+(sequence_length-1)] has
    # insert_count[b, cell] != insert_count[b, start_t], then it is invalid.
    # We'll do an all-equal check:
    def check_valid(b, start_t):
        ts = (start_t + jnp.arange(sequence_length)) % max_length_time_axis
        ref_count = state.insert_count[b, start_t]
        counts = state.insert_count[b, ts]
        # valid if all counts == ref_count
        return jnp.all(counts == ref_count)

    # vectorize over each sample
    valid_mask = jax.vmap(check_valid)(
        batch_indices, time_indices
    )  # shape [batch_size], bool

    # Mask out the invalid priorities
    priorities *= valid_mask

    # We handle the invalid draws by replacing them with the single index in this batch
    # that has the highest priority.  This is a workaround in a pure functional
    # setting (no while-loop re-sampling).
    max_idx_in_batch = jnp.argmax(priorities)
    fallback_sampled_index = sampled_item_indices[max_idx_in_batch]
    fallback_priority = priorities[max_idx_in_batch]
    final_sampled_indices = jnp.where(
        valid_mask, sampled_item_indices, fallback_sampled_index
    )
    final_priorities = jnp.where(valid_mask, priorities, fallback_priority)

    # Get the actual trajectories
    final_trajectories = _get_sample_trajectories(
        final_sampled_indices, max_length_time_axis, period, sequence_length, state
    )

    return PrioritisedTrajectoryBufferSample(
        experience=final_trajectories,
        indices=final_sampled_indices,
        priorities=final_priorities,
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
