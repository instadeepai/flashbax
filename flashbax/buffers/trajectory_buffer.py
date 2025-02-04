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

    return TrajectoryBufferState(
        experience=experience,
        is_full=jnp.array(False, dtype=bool),
        current_index=jnp.array(0),
    )


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
    # Check that the batch has the correct shape and dtypes.
    chex.assert_tree_shape_prefix(batch, utils.get_tree_shape_prefix(state.experience))
    chex.assert_trees_all_equal_dtypes(batch, state.experience)

    # Get the length of the time axis of the buffer state.
    max_length_time_axis = utils.get_tree_shape_prefix(state.experience, n_axes=2)[1]
    # Check that the sequence length is less than or equal the maximum length of the time axis.
    chex.assert_axis_dimension_lteq(
        jax.tree_util.tree_leaves(batch)[0], 1, max_length_time_axis
    )
    # Determine how many timesteps are in this batch.
    seq_len = utils.get_tree_shape_prefix(batch, n_axes=2)[1]
    # Compute the time indices where the new data will be written.
    indices = (jnp.arange(seq_len) + state.current_index) % max_length_time_axis

    # Update the buffer state.
    new_experience = jax.tree_map(
        lambda exp_field, batch_field: exp_field.at[:, indices].set(batch_field),
        state.experience,
        batch,
    )

    new_current_index = state.current_index + seq_len
    new_is_full = state.is_full | (new_current_index >= max_length_time_axis)
    new_current_index = new_current_index % max_length_time_axis

    return state.replace(  # type: ignore
        experience=new_experience,
        current_index=new_current_index,
        is_full=new_is_full,
    )


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

    # Get add_batch_size and the full size of the time axis.
    add_batch_size, max_length_time_axis = utils.get_tree_shape_prefix(
        state.experience, n_axes=2
    )

    # When full, the max time index is max_length_time_axis otherwise it is current index.
    max_time = jnp.where(state.is_full, max_length_time_axis, state.current_index)
    # When full, the oldest valid data is current_index otherwise it is zero.
    head = jnp.where(state.is_full, state.current_index, 0)

    # Given no wrap around, the last valid starting index is:
    max_start = max_time - sequence_length
    # If max_start is negative then we cannot sample yet.
    # Otherwise the number of valid items in the buffer are (max_start // period) + 1.
    num_valid_items = jnp.where(max_start >= 0, (max_start // period) + 1, 0)
    # (num_valid_items is the number of candidate subsequences—each starting at a
    # multiple of period that lie entirely in the valid region.)

    # Split the RNG key for sampling items and batch indices.
    rng_key, subkey_items = jax.random.split(rng_key)
    rng_key, subkey_batch = jax.random.split(rng_key)

    # Sample an item index in [0, num_valid_items). (This is the index in the candidate list.)
    sampled_item_idx = jax.random.randint(
        subkey_items, (batch_size,), 0, num_valid_items
    )
    # Compute the logical start time index: ls = (sampled_item_idx * period).
    logical_start = sampled_item_idx * period
    # Map logical time to physical index in the buffer given there is wrap around.
    physical_start = (head + logical_start) % max_length_time_axis

    # Also sample which add_batch row to use.
    sampled_batch_indices = jax.random.randint(
        subkey_batch, (batch_size,), 0, add_batch_size
    )
    # Create indices for the full subsequence.
    traj_time_indices = (
        physical_start[:, None] + jnp.arange(sequence_length)
    ) % max_length_time_axis

    batch_trajectory = jax.tree_map(
        lambda x: x[sampled_batch_indices[:, None], traj_time_indices],
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

    assert max_length_time_axis is not None
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
