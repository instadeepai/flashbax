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


""""Pure functions defining the trajectory buffer. The trajectory buffer takes batches of n-step
experience data, where n is the number of time steps within a trajectory. The trajectory buffer
concatenates consecutive batches of experience data along the time axis, retaining their ordering.
This allows for random sampling of the trajectories within the buffer.
"""

import functools
import warnings
from typing import TYPE_CHECKING, Callable, Generic, Optional, TypeVar

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass

import chex
import jax
import jax.numpy as jnp
from jax import Array

from flashbax import utils

Experience = TypeVar("Experience", bound=chex.ArrayTree)


@dataclass(frozen=True)
class TrajectoryBufferState(Generic[Experience]):
    """State of the  trajectory replay buffer.

    Attributes:
        experience: Arbitrary pytree containing the experience data, for example a single
            timestep (s,a,r). These are stacked along the first axis.
        current_index: Index where the next batch of experience data will be added to.
        is_full: Whether the buffer state is completely full with experience (otherwise it will
            have some empty padded values).
    """

    experience: Experience
    current_index: Array
    is_full: Array


@dataclass(frozen=True)
class TrajectoryBufferSample(Generic[Experience]):
    """Container for samples from the buffer

    Attributes:
        experience: Arbitrary pytree containing a batch of experience data.
    """

    experience: Experience


def init(
    experience: Experience,
    add_batch_size: int,
    max_length_time_axis: int,
) -> TrajectoryBufferState[Experience]:
    """
    Initialise the buffer state.

    Args:
        experience: A single timestep (e.g. (s,a,r)) used for inferring
            the structure of the experience data that will be saved in the buffer state.
        add_batch_size: Batch size of experience added to the buffer's state using the `add`
            function. I.e. the leading batch size of added experience should have size
            `add_batch_size`.
        max_length_time_axis: Maximum length of the buffer along the time axis (second axis of the
            experience data).

    Returns:
        state: Initial state of the replay buffer. All values are empty as no experience has
            been added yet.
    """
    # Set experience value to be empty.
    experience = jax.tree_map(jnp.empty_like, experience)

    # Broadcast to [add_batch_size, max_length_time_axis]
    experience = jax.tree_map(
        lambda x: jnp.broadcast_to(
            x[None, None, ...], (add_batch_size, max_length_time_axis, *x.shape)
        ),
        experience,
    )

    state = TrajectoryBufferState(
        experience=experience,
        is_full=jnp.array(False, dtype=bool),
        current_index=jnp.array(0),
    )
    return state


def add(
    state: TrajectoryBufferState[Experience],
    batch: Experience,
) -> TrajectoryBufferState[Experience]:
    """
    Add a batch of experience to the buffer state. Assumes that this carries on from the episode
    where the previous added batch of experience ended. For example, if we consider a single
    trajectory within the batch; if the last timestep of the previous added trajectory's was at
    time `t` then the first timestep of the current trajectory will be at time `t + 1`.

    Args:
        state: The buffer state.
        batch: A batch of experience. The leading axis of the pytree is the batch dimension.
            This must match `add_batch_size` and the structure of the experience used
            during initialisation of the buffer state. This batch is added along the time axis of
            the buffer state.


    Returns:
        A new buffer state with the batch of experience added.
    """
    # Check that the batch has the correct shape.
    chex.assert_tree_shape_prefix(batch, utils.get_tree_shape_prefix(state.experience))
    # Check that the batch has the correct dtypes.
    chex.assert_trees_all_equal_dtypes(batch, state.experience)

    # Get the length of the time axis of the buffer state.
    max_length_time_axis = utils.get_tree_shape_prefix(state.experience, n_axes=2)[1]
    # Check that the sequence length is less than or equal the maximum length of the time axis.
    chex.assert_axis_dimension_lteq(
        jax.tree_util.tree_leaves(batch)[0], 1, max_length_time_axis
    )

    # Get the length of the sequence of the batch.
    seq_len = utils.get_tree_shape_prefix(batch, n_axes=2)[1]

    # Calculate index location in the state where we will assign the batch of experience.
    indices = (jnp.arange(seq_len) + state.current_index) % max_length_time_axis

    # Update the buffer state.
    experience = jax.tree_util.tree_map(
        lambda experience_field, batch_field: experience_field.at[:, indices].set(
            batch_field
        ),
        state.experience,
        batch,
    )

    new_index = state.current_index + seq_len
    is_full = state.is_full | (new_index >= max_length_time_axis)
    new_index = new_index % max_length_time_axis

    state = state.replace(  # type: ignore
        experience=experience,
        current_index=new_index,
        is_full=is_full,
    )

    return state


def get_invalid_indices(
    state: TrajectoryBufferState[Experience],
    sample_sequence_length: int,
    period: int,
    add_batch_size: int,
    max_length_time_axis: int,
) -> Array:
    """
    Get the indices of the items that will be invalid when sampling from the buffer state. This
    is used to mask out the invalid items when sampling. The indices are in the format of a
    flattened array and refer to items, not the actual data. To convert item indices into data
    indices, we would perform the following:

        indices = item_indices * period
        row_indices = indices // max_length_time_axis
        time_indices = indices % max_length_time_axis

    Item indices essentially refer to a flattened array picture of the
    items (i.e. subsequences that can be sampled) in the buffer state.


    Args:
        state: The buffer state.
        sample_sequence_length: The length of the sequence that will be sampled from the buffer
            state.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.
        add_batch_size: The number of trajectories that will be added to the buffer state.
        max_length_time_axis: The maximum length of the time axis of the buffer state.

    Returns:
        The indices of the items (with shape : [add_batch_size, num_items]) that will be invalid
        when sampling from the buffer state.
    """
    # We get the max subsequence data index as done in the add function.
    max_divisible_length = max_length_time_axis - (max_length_time_axis % period)
    max_subsequence_data_index = max_divisible_length - 1
    # We get the data index that is at least sample_sequence_length away from the
    # current index.
    previous_valid_data_index = (
        state.current_index - sample_sequence_length
    ) % max_length_time_axis
    # We ensure that this index is not above the maximum mappable data index of the buffer.
    previous_valid_data_index = jnp.minimum(
        previous_valid_data_index, max_subsequence_data_index
    )
    # We then convert the data index into the item index and add one to get the index
    # of the item that is broken apart.
    invalid_item_starting_index = (previous_valid_data_index // period) + 1
    # We then take the modulo of the invalid item index to ensure that it is within the
    # bounds of the priority array. max_length_time_axis // period is the maximum number
    # of items/subsequences that can be sampled from the buffer state.
    invalid_item_starting_index = invalid_item_starting_index % (
        max_length_time_axis // period
    )

    # Calculate the maximum number of items/subsequences that can start within a
    # sample length of data. We add one to account for situations where the max
    # number of items has been broken. Often, this will unfortunately mask an item
    # that is valid however this should not be a severe issue as it would be only
    # one additional item.
    max_num_invalid_items = (sample_sequence_length // period) + 1
    # Get the actual indices of the items we cannot sample from.
    invalid_item_indices = (
        jnp.arange(max_num_invalid_items) + invalid_item_starting_index
    ) % (max_length_time_axis // period)
    # Since items that are broken are broken in the same place in each row, we
    # broadcast and add the total number of items to each index to reference
    # the invalid items in each add_batch row.
    invalid_item_indices = invalid_item_indices + jnp.arange(add_batch_size)[
        :, None
    ] * (max_length_time_axis // period)

    return invalid_item_indices


def calculate_uniform_item_indices(
    state: TrajectoryBufferState[Experience],
    rng_key: chex.PRNGKey,
    batch_size: int,
    sample_sequence_length: int,
    period: int,
    add_batch_size: int,
    max_length_time_axis: int,
) -> Array:
    """Randomly sample a batch of item indices from the buffer state. This is done uniformly.

    Args:
        state: The buffer's state.
        rng_key: Random key.
        batch_size: Batch size of sampled experience.
        sample_sequence_length: Length of trajectory to sample.
        period: The period refers to the interval between sampled sequences. It serves to regulate
            how much overlap there is between the trajectories that are sampled. To understand the
            degree of overlap, you can calculate it as the difference between the
            sample_sequence_length and the period. For instance, if you set period=1, it means that
            trajectories will be sampled uniformly with the potential for any degree of overlap. On
            the other hand, if period is equal to sample_sequence_length - 1, then trajectories can
            be sampled in a way where only the first and last timesteps overlap with each other.
            This helps you control the extent of overlap between consecutive sequences in your
            sampling process.
        add_batch_size: The number of trajectories that will be added to the buffer state.
        max_length_time_axis: The maximum length of the time axis of the buffer state.

    Returns:
        The indices of the items that will be sampled from the buffer state.

    """
    # Get the max subsequence data index to ensure we dont sample items
    # that should not ever be sampled i.e. a subsequence beyond the period
    # boundary.
    max_divisible_length = max_length_time_axis - (max_length_time_axis % period)
    max_subsequence_data_index = max_divisible_length - 1
    # Get the maximum valid time index of the data buffer based on
    # whether it is full or not.
    max_data_time_index = jnp.where(
        state.is_full,
        max_subsequence_data_index,
        state.current_index - sample_sequence_length,
    )
    # Convert the max time index to the maximum non-valid item index. This is the item
    # index that we can sample up to (excluding). We add 1 since the max time index is the last
    # valid time index that we can sample from and we want the exclusive upper bound
    # or in the case of a full buffer, the size of one row of the item array.
    max_item_time_index = (max_data_time_index // period) + 1

    # Get the indices of the items that will be invalid when sampling.
    invalid_item_indices = get_invalid_indices(
        state=state,
        sample_sequence_length=sample_sequence_length,
        period=period,
        add_batch_size=add_batch_size,
        max_length_time_axis=max_length_time_axis,
    )
    # Since all the invalid indices are repeated albeit with a batch offset,
    # we can just take the first row of the invalid indices for calculation.
    invalid_item_indices = invalid_item_indices[0]

    # We then get the upper bound of the item indices that we can sample from.
    # When being initially populated with data, the max time index will already account
    # for the items that cannot be sampled meaning that invalid indices are not needed.
    # Additionally, there is separate logic that needs to be performed when the buffer is not full.
    # When the buffer is full, the max time index will not account for the items that cannot be
    # sampled meaning that we need to subtract the number of invalid items from the
    # max item index.
    num_invalid_items = jnp.where(state.is_full, invalid_item_indices.shape[0], 0)
    upper_bound = max_item_time_index - num_invalid_items

    # Since the invalid item indices are always consecutive (in a circular manner),
    # we can get the offset by taking the last item index and adding one.
    time_offset = invalid_item_indices[-1] + 1

    # We then sample a batch of item indices over the time axis.
    sampled_item_time_indices = jax.random.randint(
        rng_key, (batch_size,), 0, upper_bound
    )
    # We then add the offset and modulo the indices to ensure that they are within
    # the bounds of the item array (which doesnt actually exist). We modulo by the
    # max item index to ensure that we loop back to the start of the item array.
    sampled_item_time_indices = (
        sampled_item_time_indices + time_offset
    ) % max_item_time_index

    # We then get the batch indices by sampling a batch of indices over the batch axis.
    sampled_item_batch_indices = jax.random.randint(
        rng_key, (batch_size,), 0, add_batch_size
    )

    # We then calculate the item indices by multiplying the batch indices by the
    # number of items in each batch and adding the time indices. This gives us
    # a flattened array picture of the items we will sample from.
    item_indices = (
        sampled_item_batch_indices * (max_length_time_axis // period)
    ) + sampled_item_time_indices

    return item_indices


def sample(
    state: TrajectoryBufferState[Experience],
    rng_key: chex.PRNGKey,
    batch_size: int,
    sequence_length: int,
    period: int,
) -> TrajectoryBufferSample[Experience]:
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
    # Calculate the indices of the items that will be sampled.
    item_indices = calculate_uniform_item_indices(
        state,
        rng_key,
        batch_size,
        sequence_length,
        period,
        add_batch_size,
        max_length_time_axis,
    )

    # Convert the item indices to the indices of the data buffer
    flat_data_indices = item_indices * period
    # Get the batch index and time index of the sampled items.
    batch_data_indices = flat_data_indices // max_length_time_axis
    time_data_indices = flat_data_indices % max_length_time_axis

    # The buffer is circular, so we can loop back to the start (`% max_length_time_axis`)
    # if the time index is greater than the length. We then add the sequence length to get
    # the end index of the sequence.
    time_data_indices = (
        jnp.arange(sequence_length) + time_data_indices[:, jnp.newaxis]
    ) % max_length_time_axis

    # Slice the experience in the buffer to get a batch of trajectories of length sequence_length
    batch_trajectory = jax.tree_util.tree_map(
        lambda x: x[batch_data_indices[:, jnp.newaxis], time_data_indices],
        state.experience,
    )

    return TrajectoryBufferSample(experience=batch_trajectory)


def can_sample(
    state: TrajectoryBufferState[Experience], min_length_time_axis: int
) -> Array:
    """Indicates whether the buffer has been filled above the minimum length, such that it
    may be sampled from."""
    return state.is_full | (state.current_index >= min_length_time_axis)


BufferState = TypeVar("BufferState", bound=TrajectoryBufferState)
BufferSample = TypeVar("BufferSample", bound=TrajectoryBufferSample)


@dataclass(frozen=True)
class TrajectoryBuffer(Generic[Experience, BufferState, BufferSample]):
    """Pure functions defining the trajectory buffer. This buffer assumes batches added to the
    buffer are a pytree with a shape prefix of (batch_size, trajectory_length). Consecutive batches
    are then concatenated along the second axis (i.e. the time axis). During sampling this allows
    for trajectories to be sampled - by slicing consecutive sequences along the time axis.

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

    See `make_trajectory_buffer` for how this container is instantiated.
    """

    init: Callable[[Experience], BufferState]
    add: Callable[
        [BufferState, Experience],
        BufferState,
    ]
    sample: Callable[
        [BufferState, chex.PRNGKey],
        BufferSample,
    ]
    can_sample: Callable[[BufferState], Array]


def validate_size(
    max_length_time_axis: Optional[int], max_size: Optional[int], add_batch_size: int
) -> None:
    if max_size is not None and max_length_time_axis is not None:
        raise ValueError(
            "Cannot specify both `max_size` and `max_length_time_axis` arguments."
        )
    if max_size is not None:
        warnings.warn(
            "Setting max_size dynamically sets the `max_length_time_axis` to "
            f"be `max_size`//`add_batch_size = {max_size // add_batch_size}`."
            "This allows one to control exactly how many timesteps are stored in the buffer."
            "Note that this overrides the `max_length_time_axis` argument.",
            stacklevel=1,
        )


def validate_trajectory_buffer_args(
    max_length_time_axis: Optional[int],
    min_length_time_axis: int,
    add_batch_size: int,
    sample_sequence_length: int,
    period: int,
    max_size: Optional[int],
) -> None:
    """Validate the arguments of the trajectory buffer."""

    validate_size(max_length_time_axis, max_size, add_batch_size)

    if max_size is not None:
        max_length_time_axis = max_size // add_batch_size

    if sample_sequence_length > min_length_time_axis:
        warnings.warn(
            "`sample_sequence_length` greater than `min_length_time_axis`, therefore "
            "overriding `min_length_time_axis`"
            "to be set to `sample_sequence_length`, as we need at least `sample_sequence_length` "
            "timesteps added to the buffer before we can sample.",
            stacklevel=1,
        )
        min_length_time_axis = sample_sequence_length

    if period > sample_sequence_length:
        warnings.warn(
            "Setting period greater than sample_sequence_length will result in no overlap between"
            f"trajectories, however, {period-sample_sequence_length} transitions will "
            "never be sampled. Setting period to be equal to sample_sequence_length will "
            "also result in no overlap between trajectories, however, all transitions will "
            "be sampled. Setting period to be `sample_sequence_length - 1` is generally "
            "desired to ensure that only starting and ending transitions are shared "
            "between trajectories allowing for utilising last transitions for bootstrapping.",
            stacklevel=1,
        )

    if max_length_time_axis is not None:
        if sample_sequence_length > max_length_time_axis:
            raise ValueError(
                "`sample_sequence_length` must be less than or equal to `max_length_time_axis`."
            )

        if min_length_time_axis > max_length_time_axis:
            raise ValueError(
                "`min_length_time_axis` must be less than or equal to `max_length_time_axis`."
            )


def make_trajectory_buffer(
    add_batch_size: int,
    sample_batch_size: int,
    sample_sequence_length: int,
    period: int,
    min_length_time_axis: int,
    max_size: Optional[int] = None,
    max_length_time_axis: Optional[int] = None,
) -> TrajectoryBuffer:
    """Makes a trajectory buffer.

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

    if sample_sequence_length > min_length_time_axis:
        min_length_time_axis = sample_sequence_length

    if max_size is not None:
        max_length_time_axis = max_size // add_batch_size

    init_fn = functools.partial(
        init,
        add_batch_size=add_batch_size,
        max_length_time_axis=max_length_time_axis,
    )
    add_fn = functools.partial(
        add,
    )
    sample_fn = functools.partial(
        sample,
        batch_size=sample_batch_size,
        sequence_length=sample_sequence_length,
        period=period,
    )
    can_sample_fn = functools.partial(
        can_sample, min_length_time_axis=min_length_time_axis
    )

    return TrajectoryBuffer(
        init=init_fn,
        add=add_fn,
        sample=sample_fn,
        can_sample=can_sample_fn,
    )
